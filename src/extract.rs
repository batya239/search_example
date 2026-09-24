use std::collections::BTreeMap;

use tree_sitter::{Node, Parser, TreeCursor};

use crate::languages::Grammar;

/// Parses `source` with `grammar` and returns every identifier-like token
/// found (see `Grammar::identifier_kinds`), mapped to how many times it
/// occurred in this file -- the term frequency a future BM25 searcher needs,
/// per DESIGN.md's "Storage" section.
pub fn extract_identifiers(grammar: Grammar, source: &[u8]) -> BTreeMap<String, u32> {
    let mut parser = Parser::new();
    parser
        .set_language(&grammar.language())
        .expect("bundled grammar should always be a valid tree-sitter language");

    let Some(tree) = parser.parse(source, None) else {
        return BTreeMap::new();
    };

    let mut terms = BTreeMap::new();
    if grammar.uses_structural_extraction() {
        collect_json_keys(tree.root_node(), source, &mut terms);
    } else {
        let kinds = grammar.identifier_kinds();
        let mut cursor = tree.root_node().walk();
        walk_preorder(&mut cursor, |node| {
            if kinds.contains(&node.kind())
                && let Ok(text) = node.utf8_text(source)
            {
                *terms.entry(text.to_string()).or_insert(0) += 1;
            }
        });
    }
    terms
}

/// Plain word-tokenizer used for `.txt`/`.md`, which have no tree-sitter
/// "identifier" concept: splits on runs of non-alphanumeric/underscore
/// characters and counts each resulting word's occurrences.
pub fn extract_identifiers_fallback(text: &str) -> BTreeMap<String, u32> {
    let mut terms = BTreeMap::new();
    for word in text.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if !word.is_empty() {
            *terms.entry(word.to_string()).or_insert(0) += 1;
        }
    }
    terms
}

/// Depth-first preorder walk over every node (named and anonymous) in the
/// tree rooted at the cursor's current node.
fn walk_preorder<'tree>(cursor: &mut TreeCursor<'tree>, mut visit: impl FnMut(Node<'tree>)) {
    loop {
        visit(cursor.node());
        if cursor.goto_first_child() {
            continue;
        }
        loop {
            if cursor.goto_next_sibling() {
                break;
            }
            if !cursor.goto_parent() {
                return;
            }
        }
    }
}

/// JSON has no identifier-kind nodes: object keys are `string` nodes,
/// indistinguishable by kind from string *values*. So instead of a flat
/// kind allowlist, this walks every `pair` node and pulls the text out of
/// its `key` field specifically, concatenating `string_content` children
/// to handle keys containing escape sequences (e.g. `"foo\nbar"` splits
/// into multiple `string_content` children around an `escape_sequence`).
fn collect_json_keys(root: Node, source: &[u8], terms: &mut BTreeMap<String, u32>) {
    let mut cursor = root.walk();
    walk_preorder(&mut cursor, |node| {
        if node.kind() != "pair" {
            return;
        }
        let Some(key_node) = node.child_by_field_name("key") else {
            return;
        };
        let mut key_cursor = key_node.walk();
        let mut key_text = String::new();
        for child in key_node.named_children(&mut key_cursor) {
            if child.kind() == "string_content"
                && let Ok(text) = child.utf8_text(source)
            {
                key_text.push_str(text);
            }
        }
        if !key_text.is_empty() {
            *terms.entry(key_text).or_insert(0) += 1;
        }
    });
}
