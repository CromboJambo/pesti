//! Prompt template families for LLM chat formatting.
//!
//! Provides ChatML, Llama3, and OpenChat template rendering with auto-routing
//! based on model name.
//!
//! ## Template Families
//!
//! - **ChatML**: `<|im_start|>role\ncontent<|im_end|>` style
//! - **Llama3**: `<|start_header_id|>role<|end_header_id|>\ncontent<|eot_id|>` style
//! - **OpenChat**: `role: content\n` style
//!
//! ## Auto-routing
//!
//! `infer_template()` classifies model names:
//! - Llama 3 variants → Llama3
//! - Everything else → ChatML (default)

use serde::{Deserialize, Serialize};

/// Chat template family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemplateFamily {
    ChatML,
    Llama3,
    OpenChat,
}

impl TemplateFamily {
    /// Render a prompt from system prompt, messages, and optional input.
    pub fn render(
        &self,
        system: Option<&str>,
        messages: &[(String, String)],
        input: Option<&str>,
    ) -> String {
        match self {
            TemplateFamily::ChatML => self.render_chatml(system, messages, input),
            TemplateFamily::Llama3 => self.render_llama3(system, messages, input),
            TemplateFamily::OpenChat => self.render_openchat(messages, input),
        }
    }

    /// Get stop tokens for this template family.
    pub fn stop_tokens(&self) -> Vec<String> {
        match self {
            TemplateFamily::ChatML => vec!["<|im_end|>".to_string(), "<|im_start|>".to_string()],
            TemplateFamily::Llama3 => {
                vec!["<|eot_id|>".to_string(), "<|end_of_text|>".to_string()]
            }
            TemplateFamily::OpenChat => vec![],
        }
    }

    fn render_chatml(
        &self,
        system: Option<&str>,
        messages: &[(String, String)],
        input: Option<&str>,
    ) -> String {
        let mut s = String::new();
        if let Some(sys) = system {
            s.push_str(&format!("<|im_start|>system\n{}<|im_end|>\n", sys));
        }
        for (role, content) in messages {
            s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, content));
        }
        if let Some(inp) = input {
            s.push_str(&format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                inp
            ));
        }
        s
    }

    fn render_llama3(
        &self,
        system: Option<&str>,
        messages: &[(String, String)],
        input: Option<&str>,
    ) -> String {
        let mut s = String::new();
        s.push_str("<|begin_of_text|>");
        if let Some(sys) = system {
            s.push_str(&format!(
                "<|start_header_id|>system<|end_header_id|>\n{}<|eot_id|>",
                sys
            ));
        }
        for (role, content) in messages {
            s.push_str(&format!(
                "<|start_header_id|>{}<|end_header_id|>\n{}<|eot_id|>",
                role, content
            ));
        }
        if let Some(inp) = input {
            s.push_str(&format!(
                "<|start_header_id|>user<|end_header_id|>\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n",
                inp
            ));
        }
        s
    }

    fn render_openchat(&self, messages: &[(String, String)], input: Option<&str>) -> String {
        let mut s = String::new();
        for (role, content) in messages {
            s.push_str(&format!("{}: {}\n", role, content));
        }
        if let Some(inp) = input {
            s.push_str(&format!("user: {}\nassistant: ", inp));
        } else {
            s.push_str("assistant: ");
        }
        s
    }
}

/// Infer template family from a model name.
///
/// - Llama 3 variants (llama-3, llama3, meta-llama-3) → Llama3
/// - Everything else → ChatML
pub fn infer_template(model_name: &str) -> TemplateFamily {
    let name_lower = model_name.to_lowercase();
    if name_lower.contains("llama-3")
        || name_lower.contains("llama3")
        || name_lower.contains("meta-llama-3")
    {
        TemplateFamily::Llama3
    } else {
        TemplateFamily::ChatML
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ChatML ─────────────────────────────────────────────────────────

    #[test]
    fn chatml_render_basic() {
        let template = TemplateFamily::ChatML;
        let messages = vec![("user".to_string(), "Hello".to_string())];
        let result = template.render(None, &messages, None);
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("Hello"));
        assert!(result.contains("<|im_end|>"));
    }

    #[test]
    fn chatml_render_with_system() {
        let template = TemplateFamily::ChatML;
        let messages = vec![("user".to_string(), "Hello".to_string())];
        let result = template.render(Some("You are helpful"), &messages, None);
        assert!(result.contains("<|im_start|>system\nYou are helpful<|im_end|>"));
    }

    #[test]
    fn chatml_render_with_input() {
        let template = TemplateFamily::ChatML;
        let messages = vec![("user".to_string(), "Hello".to_string())];
        let result = template.render(None, &messages, Some("New question"));
        assert!(result.contains("<|im_start|>user\nNew question<|im_end|>"));
        assert!(result.contains("<|im_start|>assistant\n"));
    }

    #[test]
    fn chatml_render_full_conversation() {
        let template = TemplateFamily::ChatML;
        let messages = vec![
            ("user".to_string(), "Hi".to_string()),
            ("assistant".to_string(), "Hello!".to_string()),
        ];
        let result = template.render(Some("Be nice"), &messages, Some("How are you?"));
        assert!(result.contains("<|im_start|>system"));
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("<|im_start|>assistant"));
        assert!(result.contains("How are you?"));
    }

    #[test]
    fn chatml_stop_tokens() {
        let template = TemplateFamily::ChatML;
        let stop_tokens = template.stop_tokens();
        assert_eq!(stop_tokens.len(), 2);
        assert!(stop_tokens.contains(&"<|im_end|>".to_string()));
        assert!(stop_tokens.contains(&"<|im_start|>".to_string()));
    }

    // ── Llama3 ─────────────────────────────────────────────────────────

    #[test]
    fn llama3_render_basic() {
        let template = TemplateFamily::Llama3;
        let messages = vec![("user".to_string(), "Test".to_string())];
        let result = template.render(None, &messages, None);
        assert!(result.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(result.contains("Test"));
        assert!(result.contains("<|eot_id|>"));
    }

    #[test]
    fn llama3_render_with_system() {
        let template = TemplateFamily::Llama3;
        let messages = vec![("user".to_string(), "Hi".to_string())];
        let result = template.render(Some("Be concise"), &messages, None);
        assert!(result.contains("<|begin_of_text|>"));
        assert!(result.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(result.contains("Be concise<|eot_id|>"));
    }

    #[test]
    fn llama3_render_with_input() {
        let template = TemplateFamily::Llama3;
        let messages = vec![("user".to_string(), "Hi".to_string())];
        let result = template.render(None, &messages, Some("Ask me anything"));
        assert!(result.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(result.contains("Ask me anything<|eot_id|>"));
        assert!(result.contains("<|start_header_id|>assistant<|end_header_id|>"));
    }

    #[test]
    fn llama3_stop_tokens() {
        let template = TemplateFamily::Llama3;
        let stop_tokens = template.stop_tokens();
        assert_eq!(stop_tokens.len(), 2);
        assert!(stop_tokens.contains(&"<|eot_id|>".to_string()));
        assert!(stop_tokens.contains(&"<|end_of_text|>".to_string()));
    }

    // ── OpenChat ───────────────────────────────────────────────────────

    #[test]
    fn openchat_render_basic() {
        let template = TemplateFamily::OpenChat;
        let messages = vec![("user".to_string(), "Hi".to_string())];
        let result = template.render(None, &messages, None);
        assert!(result.contains("user: Hi"));
        assert!(result.contains("assistant: "));
    }

    #[test]
    fn openchat_render_with_input() {
        let template = TemplateFamily::OpenChat;
        let messages = vec![("user".to_string(), "Hello".to_string())];
        let result = template.render(None, &messages, Some("Final question"));
        assert!(result.contains("user: Final question\nassistant: "));
    }

    #[test]
    fn openchat_render_no_input() {
        let template = TemplateFamily::OpenChat;
        let messages: Vec<(String, String)> = vec![];
        let result = template.render(None, &messages, None);
        assert_eq!(result, "assistant: ");
    }

    #[test]
    fn openchat_render_full_conversation() {
        let template = TemplateFamily::OpenChat;
        let messages = vec![
            ("user".to_string(), "Hello".to_string()),
            ("assistant".to_string(), "Hi there!".to_string()),
        ];
        let result = template.render(None, &messages, Some("How are you?"));
        assert!(result.contains("user: Hello"));
        assert!(result.contains("assistant: Hi there!"));
        assert!(result.contains("user: How are you?\nassistant: "));
    }

    #[test]
    fn openchat_stop_tokens_empty() {
        let template = TemplateFamily::OpenChat;
        assert!(template.stop_tokens().is_empty());
    }

    // ── infer_template ─────────────────────────────────────────────────

    #[test]
    fn infer_llama3_variants() {
        assert!(matches!(
            infer_template("llama-3-8b"),
            TemplateFamily::Llama3
        ));
        assert!(matches!(
            infer_template("llama3-70b"),
            TemplateFamily::Llama3
        ));
        assert!(matches!(
            infer_template("meta-llama-3-instruct"),
            TemplateFamily::Llama3
        ));
        assert!(matches!(
            infer_template("Meta-Llama-3.1-8B"),
            TemplateFamily::Llama3
        ));
    }

    #[test]
    fn infer_chatml_fallback() {
        assert!(matches!(
            infer_template("tinyllama-1.1b"),
            TemplateFamily::ChatML
        ));
        assert!(matches!(
            infer_template("mistral-7b"),
            TemplateFamily::ChatML
        ));
        assert!(matches!(
            infer_template("phi-3-mini"),
            TemplateFamily::ChatML
        ));
        assert!(matches!(infer_template("qwen2-7b"), TemplateFamily::ChatML));
        assert!(matches!(
            infer_template("llama-2-7b"),
            TemplateFamily::ChatML
        ));
    }

    #[test]
    fn infer_unknown_model_defaults_chatml() {
        assert!(matches!(
            infer_template("unknown-model"),
            TemplateFamily::ChatML
        ));
        assert!(matches!(
            infer_template("custom-7b"),
            TemplateFamily::ChatML
        ));
    }

    // ── Serialization ──────────────────────────────────────────────────

    #[test]
    fn template_family_serializes_chatml() {
        let json = serde_json::to_string(&TemplateFamily::ChatML).unwrap();
        assert!(json.contains("ChatML"));
        let parsed: TemplateFamily = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, TemplateFamily::ChatML));
    }

    #[test]
    fn template_family_serializes_llama3() {
        let json = serde_json::to_string(&TemplateFamily::Llama3).unwrap();
        assert!(json.contains("Llama3"));
        let parsed: TemplateFamily = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, TemplateFamily::Llama3));
    }

    #[test]
    fn template_family_serializes_openchat() {
        let json = serde_json::to_string(&TemplateFamily::OpenChat).unwrap();
        assert!(json.contains("OpenChat"));
        let parsed: TemplateFamily = serde_json::from_str(&json).unwrap();
        assert!(matches!(parsed, TemplateFamily::OpenChat));
    }

    // ── Edge Cases ─────────────────────────────────────────────────────

    #[test]
    fn chatml_empty_messages() {
        let template = TemplateFamily::ChatML;
        let result = template.render(None, &[], None);
        assert!(result.is_empty());
    }

    #[test]
    fn llama3_empty_messages() {
        let template = TemplateFamily::Llama3;
        let result = template.render(None, &[], None);
        assert!(result.starts_with("<|begin_of_text|>"));
    }

    #[test]
    fn openchat_empty_messages_no_input() {
        let template = TemplateFamily::OpenChat;
        let result = template.render(None, &[], None);
        assert_eq!(result, "assistant: ");
    }
}
