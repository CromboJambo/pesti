//! Mutability annotation layer.
//! 
//! Tracks mutability state through the AST: let vs let mut, &self vs &mut self,
//! and mutation sites.

use std::collections::{HashMap, HashSet};

/// Mutability state of a binding or reference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutability {
    Immutable,   // let x = ...; &T; fn foo(&self)
    Mutable,     // let mut x = ...; &mut T; fn foo(&mut self)
}

/// Kind of mutability annotation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutKind {
    Binding,      // let/let mut variable declaration
    Reference,    // & or &mut reference
    MethodSelf,   // &self vs &mut self receiver
    FieldAccess,  // mutation through field access
}

/// Annotation for mutability at a node
#[derive(Debug, Clone)]
pub struct MutabilityAnnotation {
    pub mutable: bool,
    /// What kind of mutability this is
    pub kind: MutKind,
    pub confidence: u8,
}

/// Tracks mutability state through the AST
pub struct MutabilityAnalyzer {
    /// Current scope's mutable bindings: name -> is_mutable
    scopes: Vec<HashMap<String, bool>>,
    
    pub annotations: HashMap<usize, MutabilityAnnotation>,
}

impl MutabilityAnalyzer {
    pub fn new() -> Self {
        let mut scopes = Vec::new();
        scopes.push(HashMap::new());
        Self {
            scopes,
            annotations: HashMap::new(),
        }
    }

    fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn exit_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    fn bind(&mut self, name: &str, mutable: bool, node_id: usize, kind: MutKind) {
        let scope = self.scopes.last_mut().unwrap();
        scope.insert(name.to_string(), mutable);
        
        self.annotations.insert(node_id, MutabilityAnnotation {
            mutable,
            kind,
            confidence: 2,  // proven from syntax
        });
    }

    fn lookup(&self, name: &str) -> Option<bool> {
        for scope in self.scopes.iter().rev() {
            if let Some(mutable) = scope.get(name) {
                return Some(*mutable);
            }
        }
        None
    }

    /// Analyze function parameters for mutability
    fn analyze_params(&mut self, params: &[&str]) {
        for (i, param) in params.iter().enumerate() {
            let mutable = if param.starts_with("&mut ") {
                true
            } else {
                false
            };
            
            // Bind with reference mutability kind
            let name = param.trim_start_matches('&').trim_start_matches(' ').trim_end();
            self.bind(name, mutable, i, MutKind::Reference);
        }
    }

    /// Analyze method receiver (&self vs &mut self)
    fn analyze_method(&mut self, func: &str, node_id: usize) {
        if func.contains("&mut self") {
            self.annotations.insert(node_id, MutabilityAnnotation {
                mutable: true,
                kind: MutKind::MethodSelf,
                confidence: 2,
            });
        } else if func.contains("&self") {
            self.annotations.insert(node_id, MutabilityAnnotation {
                mutable: false,
                kind: MutKind::MethodSelf,
                confidence: 2,
            });
        }
    }

    /// Main analysis entry for a function
    pub fn analyze_function(&mut self, func_sig: &str, body: &str) {
        // Extract params from signature (simplified)
        let params: Vec<&str> = vec![];  // Would parse real params here
        
        self.analyze_params(&params);
        self.analyze_method(func_sig, 0);
        
        self.enter_scope();
        
        // Analyze body for let bindings and mutations
        for (i, line) in body.lines().enumerate() {
            if line.trim_start().starts_with("let mut ") {
                // Extract variable name (simplified)
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let name = parts[2].trim_end_matches(' ');
                    self.bind(name, true, i + 100, MutKind::Binding);
                }
            } else if line.trim_start().starts_with("let ") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let name = parts[2].trim_end_matches(' ');
                    self.bind(name, false, i + 100, MutKind::Binding);
                }
            }
        }
        
        self.exit_scope();
    }
}

/// Analyze mutability across an entire program
pub fn analyze_mutability(source: &str) -> HashMap<usize, MutabilityAnnotation> {
    let mut analyzer = MutabilityAnalyzer::new();
    
    // Simplified: analyze whole source as one function
    analyzer.analyze_function("", source);
    
    analyzer.annotations
}
