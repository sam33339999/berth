//! Think-block split, matching `serve_openai.py` `ThinkStream` and `_chat`.

const THINK_CLOSE: &str = "</think>";

#[derive(Debug)]
pub struct ThinkStream {
    in_think: bool,
    pending: String,
    skip_ws: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ThinkPart {
    Reasoning(String),
    Content(String),
}

impl ThinkStream {
    pub fn new(in_think: bool) -> Self {
        Self {
            in_think,
            pending: String::new(),
            skip_ws: false,
        }
    }

    pub fn in_think(&self) -> bool {
        self.in_think
    }

    pub fn feed(&mut self, text: &str) -> Vec<ThinkPart> {
        if text.is_empty() {
            return Vec::new();
        }
        if !self.in_think {
            return self.content(text);
        }
        self.pending.push_str(text);
        if let Some(idx) = self.pending.find(THINK_CLOSE) {
            let reasoning = self.pending[..idx].to_string();
            let rest = self.pending[idx + THINK_CLOSE.len()..].to_string();
            self.pending.clear();
            self.in_think = false;
            self.skip_ws = true;
            let mut out = Vec::new();
            if !reasoning.is_empty() {
                out.push(ThinkPart::Reasoning(reasoning));
            }
            out.extend(self.content(&rest));
            return out;
        }
        let mut hold = 0;
        for n in 1..THINK_CLOSE.len() {
            if self.pending.ends_with(&THINK_CLOSE[..n]) {
                hold = n;
            }
        }
        let emit = if hold == 0 {
            std::mem::take(&mut self.pending)
        } else {
            let cut = self.pending.len() - hold;
            let emit = self.pending[..cut].to_string();
            self.pending.drain(..cut);
            emit
        };
        if emit.is_empty() {
            Vec::new()
        } else {
            vec![ThinkPart::Reasoning(emit)]
        }
    }

    pub fn finish(&mut self) -> Vec<ThinkPart> {
        let tail = std::mem::take(&mut self.pending);
        if tail.is_empty() {
            return Vec::new();
        }
        if self.in_think {
            vec![ThinkPart::Reasoning(tail)]
        } else {
            self.content(&tail)
        }
    }

    fn content(&mut self, text: &str) -> Vec<ThinkPart> {
        let text = if self.skip_ws {
            let stripped = text.trim_start_matches(['\n', ' ']);
            if stripped.is_empty() {
                return Vec::new();
            }
            self.skip_ws = false;
            stripped
        } else {
            text
        };
        vec![ThinkPart::Content(text.to_string())]
    }
}

/// Non-stream split from `serve_openai.py` `_chat`.
///
/// One cut on `</think>`. Leading newlines and spaces are stripped from the
/// answer. A whitespace-only answer is replaced with the full generation.
pub fn split_nonstream(text: &str) -> (Option<String>, String) {
    let Some(idx) = text.find(THINK_CLOSE) else {
        return (None, text.to_string());
    };
    let reasoning = text[..idx].to_string();
    let mut content = text[idx + THINK_CLOSE.len()..].to_string();
    content = content.trim_start_matches(['\n', ' ']).to_string();
    if content.trim().is_empty() {
        content = text.to_string();
    }
    (Some(reasoning), content)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(parts: &[ThinkPart], reasoning: &mut String, content: &mut String) {
        for part in parts {
            match part {
                ThinkPart::Reasoning(text) => reasoning.push_str(text),
                ThinkPart::Content(text) => content.push_str(text),
            }
        }
    }

    #[test]
    fn close_tag_held_across_chunks() {
        let mut stream = ThinkStream::new(true);
        let mut reasoning = String::new();
        let mut content = String::new();
        for piece in ["<think>\nwave ", "</thi", "nk>\n\n", "OK"] {
            let fed = stream.feed(piece);
            collect(&fed, &mut reasoning, &mut content);
        }
        let tail = stream.finish();
        collect(&tail, &mut reasoning, &mut content);
        assert_eq!(reasoning, "<think>\nwave ");
        assert_eq!(content, "OK");
        assert!(!stream.in_think());
    }

    #[test]
    fn whole_string_splits_like_the_serve() {
        let text = "<think>\nwave 👋\n</think>\n\nOK";
        let mut stream = ThinkStream::new(true);
        let mut reasoning = String::new();
        let mut content = String::new();
        collect(&stream.feed(text), &mut reasoning, &mut content);
        collect(&stream.finish(), &mut reasoning, &mut content);
        let (once_reasoning, once_content) = split_nonstream(text);
        assert_eq!(once_reasoning.as_deref(), Some(reasoning.as_str()));
        assert_eq!(once_content, content);
        assert_eq!(reasoning, "<think>\nwave 👋\n");
        assert_eq!(content, "OK");
    }

    #[test]
    fn whitespace_only_answer_keeps_the_full_text() {
        let text = "<think>\nonly\n</think>\n  ";
        let (reasoning, content) = split_nonstream(text);
        assert_eq!(reasoning.as_deref(), Some("<think>\nonly\n"));
        assert_eq!(content, text);
    }
}
