use std::sync::Arc;

use anthropic_wire::{ContentBlockDelta, CreateMessageResponse, StopReason, StreamEvent};
use axum::response::sse::Event;
use futures::{Stream, StreamExt, TryStreamExt};
use serde::Serialize;
use serde_json::{Value, json};
use tracing::warn;

use super::tool_bridge::{MARK, ToolBridge, safe_len};

/// Represents the data structure for streaming events in OpenAI API format
/// Contains a choices array with deltas of content
#[derive(Debug, Serialize)]
struct StreamEventData {
    choices: Vec<StreamEventDelta>,
}

impl StreamEventData {
    /// Creates a new `StreamEventData` with the given content
    ///
    /// # Arguments
    /// * `content` - The event content to include
    ///
    /// # Returns
    /// A new `StreamEventData` instance with the content wrapped in choices array
    fn new(content: EventContent) -> Self {
        Self {
            choices: vec![StreamEventDelta { delta: content }],
        }
    }
}

/// Represents a delta update in a streaming response
/// Contains the content change for the current chunk
#[derive(Debug, Serialize)]
struct StreamEventDelta {
    delta: EventContent,
}

/// Content of an event, either regular content or reasoning (thinking mode)
/// Uses untagged enum to handle different response formats
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum EventContent {
    Content { content: String },
    Reasoning { reasoning_content: String },
}

/// Creates an SSE event with the given content in OpenAI format
///
/// # Arguments
/// * `content` - The event content to include
///
/// # Returns
/// A formatted SSE Event ready to be sent to the client
pub fn build_event(content: EventContent) -> Event {
    let event = Event::default();
    let data = StreamEventData::new(content);
    event.json_data(data).unwrap()
}

/// Transforms a Claude.ai event stream into an OpenAI-compatible event stream
///
/// Extracts content from Claude events and reformats them to match OpenAI's streaming format.
/// This function processes each event in the stream, identifying the delta content type
/// (text or thinking), and converting it to the appropriate OpenAI-compatible event format.
///
/// # Arguments
/// * `s` - The input stream of Claude.ai events
///
/// # Returns
/// A stream of OpenAI-compatible SSE events
///
/// # Type Parameters
/// * `I` - The input stream type
/// * `E` - The error type for the stream
pub fn transform_stream<I, E>(s: I) -> impl Stream<Item = Result<Event, E>>
where
    I: Stream<Item = Result<eventsource_stream::Event, E>>,
{
    s.try_filter_map(async |eventsource_stream::Event { data, .. }| {
        let Ok(parsed) = serde_json::from_str::<StreamEvent>(&data) else {
            return Ok(None);
        };
        let StreamEvent::ContentBlockDelta { delta, .. } = parsed else {
            return Ok(None);
        };
        match delta {
            ContentBlockDelta::TextDelta { text } => {
                Ok(Some(build_event(EventContent::Content { content: text })))
            }
            ContentBlockDelta::ThinkingDelta { thinking } => {
                Ok(Some(build_event(EventContent::Reasoning {
                    reasoning_content: thinking,
                })))
            }
            _ => Ok(None),
        }
    })
}

pub fn transforms_json(input: &CreateMessageResponse) -> Value {
    let content = input
        .content
        .iter()
        .filter_map(|block| match block {
            anthropic_wire::ContentBlock::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect::<String>();

    let usage = input.usage.as_ref().map(|u| {
        serde_json::json!({
            "prompt_tokens": u.input_tokens,
            "completion_tokens": u.output_tokens,
            "total_tokens": u.input_tokens + u.output_tokens
        })
    });

    let finish_reason = match input.stop_reason {
        Some(StopReason::MaxTokens | StopReason::ModelContextWindowExceeded) => "length",
        Some(StopReason::ToolUse) => "tool_calls",
        Some(StopReason::Refusal) => "content_filter",
        Some(StopReason::EndTurn | StopReason::StopSequence | StopReason::PauseTurn) | None => {
            "stop"
        }
    };

    serde_json::json!({
        "id": input.id,
        "object": "chat.completion",
        "created": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        "model": input.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": finish_reason
        }],
        "usage": usage
    })
}

fn json_event(v: Value) -> Event {
    Event::default()
        .json_data(v)
        .expect("a serde_json::Value always serializes")
}

fn delta_event(delta: Value, finish_reason: Value) -> Event {
    json_event(json!({
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
    }))
}

fn split_chunks(s: &str, size: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.is_empty() {
        return vec![String::new()];
    }
    chars.chunks(size).map(|c| c.iter().collect()).collect()
}

/// Like [`transform_stream`], but turns `<tool_call>` blocks in the model output
/// into OpenAI `tool_calls` deltas. Plain text is streamed live; once a tool block
/// starts, the rest is buffered and parsed when the stream ends.
pub fn transform_stream_tools<I, E>(
    s: I,
    bridge: Arc<ToolBridge>,
) -> impl Stream<Item = Result<Event, E>>
where
    I: Stream<Item = Result<eventsource_stream::Event, E>>,
{
    async_stream::stream! {
        futures::pin_mut!(s);
        let mut full = String::new();
        let mut pending = String::new();
        let mut in_tool = false;
        while let Some(item) = s.next().await {
            let ev = match item {
                Ok(ev) => ev,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };
            let Ok(StreamEvent::ContentBlockDelta { delta, .. }) =
                serde_json::from_str::<StreamEvent>(&ev.data)
            else {
                continue;
            };
            match delta {
                ContentBlockDelta::ThinkingDelta { thinking } => {
                    yield Ok(build_event(EventContent::Reasoning { reasoning_content: thinking }));
                }
                ContentBlockDelta::TextDelta { text } => {
                    full.push_str(&text);
                    if in_tool {
                        continue;
                    }
                    pending.push_str(&text);
                    if let Some(i) = pending.find(MARK) {
                        if i > 0 {
                            yield Ok(build_event(EventContent::Content {
                                content: pending[..i].to_string(),
                            }));
                        }
                        in_tool = true;
                        pending.clear();
                        continue;
                    }
                    let safe = safe_len(&pending);
                    if safe > 0 {
                        let out: String = pending.drain(..safe).collect();
                        yield Ok(build_event(EventContent::Content { content: out }));
                    }
                }
                _ => {}
            }
        }
        if !in_tool && !pending.is_empty() {
            yield Ok(build_event(EventContent::Content { content: std::mem::take(&mut pending) }));
        }
        let parsed = bridge.process(&full);
        if !parsed.errors.is_empty() {
            warn!("tool bridge: {}", parsed.errors.join("; "));
        }
        if in_tool && parsed.calls.is_empty() {
            // Malformed tool block: show it instead of silently dropping the reply.
            if let Some(i) = full.find(MARK) {
                yield Ok(build_event(EventContent::Content { content: full[i..].to_string() }));
            }
        }
        for (idx, c) in parsed.calls.iter().enumerate() {
            yield Ok(delta_event(
                json!({"tool_calls": [{
                    "index": idx, "id": c.id, "type": "function",
                    "function": {"name": c.name, "arguments": ""}
                }]}),
                Value::Null,
            ));
            for piece in split_chunks(&c.arguments, 48) {
                yield Ok(delta_event(
                    json!({"tool_calls": [{"index": idx, "function": {"arguments": piece}}]}),
                    Value::Null,
                ));
            }
        }
        let finish = if parsed.calls.is_empty() { "stop" } else { "tool_calls" };
        yield Ok(delta_event(json!({}), json!(finish)));
        yield Ok(Event::default().data("[DONE]"));
    }
}

/// Like [`transforms_json`], but extracts `tool_calls` from the model output.
pub fn transforms_json_tools(input: &CreateMessageResponse, bridge: &ToolBridge) -> Value {
    let mut v = transforms_json(input);
    let text = v
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let parsed = bridge.process(&text);
    if !parsed.errors.is_empty() {
        warn!("tool bridge: {}", parsed.errors.join("; "));
    }
    let malformed = parsed.calls.is_empty() && text.contains(MARK);
    if let Some(msg) = v.pointer_mut("/choices/0/message").and_then(Value::as_object_mut) {
        if parsed.calls.is_empty() {
            // Keep the original text when nothing could be parsed.
            msg.insert("content".into(), json!(if malformed { text } else { parsed.visible }));
        } else {
            msg.insert(
                "content".into(),
                if parsed.visible.is_empty() { Value::Null } else { json!(parsed.visible) },
            );
            msg.insert(
                "tool_calls".into(),
                Value::Array(parsed.calls.iter().map(|c| c.to_json()).collect()),
            );
        }
    }
    if !parsed.calls.is_empty()
        && let Some(f) = v.pointer_mut("/choices/0/finish_reason")
    {
        *f = json!("tool_calls");
    }
    v
}

#[cfg(test)]
mod tool_tests {
    use std::convert::Infallible;

    use axum::{
        body::to_bytes,
        response::{IntoResponse, Sse},
    };
    use futures::stream;

    use super::*;
    use crate::middleware::claude::tool_bridge;

    fn bridge() -> Arc<ToolBridge> {
        let mut body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "write_file", "parameters": {
                "type": "object", "required": ["path", "content"],
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}}]
        });
        Arc::new(tool_bridge::preprocess(&mut body).expect("bridge"))
    }

    fn ev(text: &str) -> Result<eventsource_stream::Event, Infallible> {
        let data = json!({
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": text}
        })
        .to_string();
        Ok(eventsource_stream::Event {
            event: "content_block_delta".into(),
            data,
            id: String::new(),
            retry: None,
        })
    }

    async fn run(chunks: &[&str]) -> Vec<Value> {
        let events: Vec<_> = chunks.iter().map(|c| ev(c)).collect();
        let sse = Sse::new(transform_stream_tools(stream::iter(events), bridge())).into_response();
        let bytes = to_bytes(sse.into_body(), usize::MAX).await.unwrap();
        String::from_utf8(bytes.to_vec())
            .unwrap()
            .lines()
            .filter_map(|l| l.strip_prefix("data:"))
            .map(str::trim)
            .filter(|d| *d != "[DONE]")
            .map(|d| serde_json::from_str(d).unwrap())
            .collect()
    }

    fn text_of(chunks: &[Value]) -> String {
        chunks
            .iter()
            .filter_map(|c| c.pointer("/choices/0/delta/content").and_then(Value::as_str))
            .collect()
    }

    #[tokio::test]
    async fn streams_text_then_emits_tool_calls() {
        let out = run(&[
            "Để mình ", "ghi file.\n<tool", "_call name=\"write_", "file\">\n<path>a.py</path>\n<content>\nprint(\"hi\")\n",
            "</content>\n</tool_call>",
        ])
        .await;
        let text = text_of(&out);
        assert!(text.starts_with("Để mình ghi file."), "{text}");
        assert!(!text.contains("<tool"), "{text}");
        let args: String = out
            .iter()
            .filter_map(|c| c.pointer("/choices/0/delta/tool_calls/0/function/arguments"))
            .filter_map(Value::as_str)
            .collect();
        assert_eq!(
            serde_json::from_str::<Value>(&args).unwrap(),
            json!({"path": "a.py", "content": "print(\"hi\")"})
        );
        let first = out
            .iter()
            .find_map(|c| c.pointer("/choices/0/delta/tool_calls/0/function/name"))
            .unwrap();
        assert_eq!(first, "write_file");
        let last = out.last().unwrap();
        assert_eq!(last.pointer("/choices/0/finish_reason").unwrap(), "tool_calls");
    }

    #[tokio::test]
    async fn plain_text_passes_through() {
        let out = run(&["Xin chào <b>bạn</b> a <tool"]).await;
        assert_eq!(text_of(&out), "Xin chào <b>bạn</b> a <tool");
        assert_eq!(out.last().unwrap().pointer("/choices/0/finish_reason").unwrap(), "stop");
    }

    #[tokio::test]
    async fn malformed_block_is_shown_not_dropped() {
        let out = run(&["<tool_call name=\"nope\"><x>1</x></tool_call>"]).await;
        assert!(text_of(&out).contains("nope"));
    }

    #[test]
    fn non_stream_json_gets_tool_calls() {
        let resp: CreateMessageResponse = serde_json::from_value(json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
            "content": [{"type": "text", "text": "ok\n<tool_call name=\"write_file\"><path>p</path><content>c</content></tool_call>"}],
            "stop_reason": "end_turn", "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }))
        .unwrap();
        let v = transforms_json_tools(&resp, &bridge());
        assert_eq!(v.pointer("/choices/0/finish_reason").unwrap(), "tool_calls");
        assert_eq!(v.pointer("/choices/0/message/content").unwrap(), "ok");
        assert_eq!(v.pointer("/choices/0/message/tool_calls/0/function/name").unwrap(), "write_file");
    }
}
