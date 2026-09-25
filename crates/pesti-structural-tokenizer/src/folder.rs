use crate::{StructuralToken, TokenKind};

#[derive(Debug, Clone)]
pub enum SpanKind {
    Block,
}

#[derive(Debug, Clone)]
pub struct Span {
    pub kind: SpanKind,
    pub open_token: Option<StructuralToken>,
    pub body: Vec<SpanItem>,
    pub close_token: Option<StructuralToken>,
}

#[derive(Debug, Clone)]
pub enum SpanItem {
    Token(StructuralToken),
    Span(Box<Span>),
}

impl Span {
    fn new(kind: SpanKind) -> Self {
        Span {
            kind,
            open_token: None,
            body: vec![],
            close_token: None,
        }
    }
}

pub fn fold(tokens: Vec<StructuralToken>) -> Option<Span> {
    let mut root = Span::new(SpanKind::Block);
    // Stack of spans - top is current scope
    let mut stack = vec![root];

    for tok in tokens {
        match tok.kind {
            TokenKind::BraceOpen => {
                // Start new block
                let mut child = Span::new(SpanKind::Block);
                child.open_token = Some(tok);
                stack.push(child);
            }
            TokenKind::BraceClose => {
                // Close current block, attach to parent's body
                if let Some(mut span) = stack.pop() {
                    span.close_token = Some(tok);
                    if let Some(parent) = stack.last_mut() {
                        parent.body.push(SpanItem::Span(Box::new(span)));
                    } else {
                        // Top-level block closed - this is the root
                        return Some(span);
                    }
                }
            }
            _ => {
                // Regular token goes into current scope's body
                if let Some(current) = stack.last_mut() {
                    current.body.push(SpanItem::Token(tok));
                }
            }
        }
    }

    // If we didn't hit a top-level close, return what we have
    if stack.len() == 1 {
        Some(stack.remove(0))
    } else {
        None
    }
}
