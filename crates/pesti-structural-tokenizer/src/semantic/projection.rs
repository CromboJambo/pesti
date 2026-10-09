//! Semantic annotation projection onto token stream.
//! 
//! Encodes semantic properties (ownership, error handling, mutability) into
//! compact bit-packed tags that can be projected alongside structural tokens.

use std::collections::HashMap;

use super::errors::{analyze_errors, ErrorAnnotation, ErrorHandling};
use super::mutability::{analyze_mutability, MutabilityAnnotation};
use super::ownership::{analyze_ownership, OwnershipAnnotation, OwnershipMode};

/// Semantic feature bits packed into a single u8 per position.
/// 
/// Bit layout:
///   0-1: ownership mode (00=none, 01=owned, 10=borrowed, 11=mut_borrowed)
///   2-4: error handling (000=none, 001=?, 010=unwrap, 011=expect, 100=match)
///   5: confidence (0=inferred, 1=proven)
///   6: mutability (0=immutable, 1=mutable)
///   7: reserved
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticTag {
    pub value: u8,
}

impl SemanticTag {
    /// Encode semantic properties into a single tag byte.
    pub fn encode(
        own: Option<OwnershipMode>,
        err: Option<ErrorHandling>,
        mutable: bool,
        conf: u8,
    ) -> Self {
        let mut bits: u8 = 0;

        // Bits 0-1: ownership mode
        if let Some(mode) = own {
            match mode {
                OwnershipMode::Owned => bits |= 0b01,
                OwnershipMode::Borrowed => bits |= 0b10,
                OwnershipMode::MutablyBorrowed => bits |= 0b11,
            }
        }

        // Bits 2-4: error handling
        if let Some(handling) = err {
            let err_bits = match handling {
                ErrorHandling::Propagated => 0b001,
                ErrorHandling::Unwrapped => 0b010,
                ErrorHandling::Expected => 0b011,
                ErrorHandling::Matched => 0b100,
                ErrorHandling::Ignored => 0b101,
                _ => 0b000,
            };
            bits |= (err_bits << 2);
        }

        // Bit 5: confidence (proven vs inferred)
        if conf >= 2 {
            bits |= 0b0010_0000;
        }

        // Bit 6: mutability
        if mutable {
            bits |= 0b0100_0000;
        }

        SemanticTag { value: bits }
    }

    /// Decode ownership mode back out.
    pub fn ownership(&self) -> Option<OwnershipMode> {
        let mode_bits = self.value & 0b11;
        match mode_bits {
            0b01 => Some(OwnershipMode::Owned),
            0b10 => Some(OwnershipMode::Borrowed),
            0b11 => Some(OwnershipMode::MutablyBorrowed),
            _ => None,
        }
    }

    /// Decode error handling back out.
    pub fn error_handling(&self) -> Option<ErrorHandling> {
        let err_bits = (self.value >> 2) & 0b111;
        match err_bits {
            0b001 => Some(ErrorHandling::Propagated),
            0b010 => Some(ErrorHandling::Unwrapped),
            0b011 => Some(ErrorHandling::Expected),
            0b100 => Some(ErrorHandling::Matched),
            0b101 => Some(ErrorHandling::Ignored),
            _ => None,
        }
    }

    /// Check if proven (confidence bit set).
    pub fn is_proven(&self) -> bool {
        self.value & 0b0010_0000 != 0
    }

    /// Check if mutable.
    pub fn is_mutable(&self) -> bool {
        self.value & 0b0100_0000 != 0
    }
}

/// Complete semantic annotation set for a tokenized program.
#[derive(Debug, Clone)]
pub struct SemanticAnnotations {
    /// One tag per structural token position.
    pub tags: Vec<SemanticTag>,
    /// Raw ownership annotations (node_id -> annotation).
    pub ownership: HashMap<usize, OwnershipAnnotation>,
    /// Raw error handling annotations.
    pub errors: HashMap<usize, ErrorAnnotation>,
    /// Raw mutability annotations.
    pub mutability: HashMap<usize, MutabilityAnnotation>,
}

/// Run all semantic analyzers and project onto token positions.
pub fn annotate_all(
    source: &str,
    token_count: usize,
) -> SemanticAnnotations {
    let ownership = analyze_ownership(source);
    let errors = analyze_errors(source);
    let mutability = analyze_mutability(source);

    // Project onto token positions (simplified 1:1 mapping)
    let tags: Vec<SemanticTag> = (0..token_count)
        .map(|pos| {
            let own_ann = ownership.get(&pos);
            let err_ann = errors.get(&pos);
            let mut_ann = mutability.get(&pos);

            SemanticTag::encode(
                own_ann.map(|a| a.mode),
                err_ann.map(|a| a.handling),
                mut_ann.map_or(false, |a| a.mutable),
                own_ann
                    .map(|a| a.confidence)
                    .or_else(|| err_ann.map(|a| a.confidence))
                    .or_else(|| mut_ann.map(|a| a.confidence))
                    .unwrap_or(0),
            )
        })
        .collect();

    SemanticAnnotations {
        tags,
        ownership,
        errors,
        mutability,
    }
}

/// Project semantic annotations onto existing token stream.
pub fn project_annotations(
    tokens: Vec<usize>,
    sem: &SemanticAnnotations,
) -> Vec<(usize, SemanticTag)> {
    tokens.iter()
        .zip(sem.tags.iter())
        .map(|(&tok, tag)| (tok, *tag))
        .collect()
}
