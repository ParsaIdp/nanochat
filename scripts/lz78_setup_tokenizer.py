"""
Build and save an LZ78 tokenizer from a TSV dictionary file.

Usage:
    python scripts/lz78_setup_tokenizer.py --tsv_path PATH --tsv_format {lz78,compressed} --output_dir DIR

Outputs:
    output_dir/
        lz78_config.json      - tokenizer config
        lz78_codes.tsv         - code -> hex bytes mapping
        token_bytes.pt         - byte length per token (for BPB eval)
        token_metadata.pt      - (parent_code, char_byte) per token (for structured embedding)
        token_metadata_hier.pt - hierarchical metadata (compressed trie only)
"""
from __future__ import annotations

import argparse
import random

from nanochat.lz78_tokenizer import LZ78Tokenizer


def verify_roundtrip(tokenizer: "LZ78Tokenizer", texts: list[str]) -> bool:
    """Verify encode/decode roundtrip on sample texts."""
    print("\n--- Roundtrip verification ---")
    all_ok = True
    for text in texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        ok = decoded == text
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {repr(text[:60])} -> {len(ids)} tokens -> {repr(decoded[:60])}")
    if all_ok:
        print("All roundtrip tests passed!")
    else:
        print("WARNING: Some roundtrip tests failed!")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LZ78 tokenizer from TSV dictionary")
    parser.add_argument("--tsv_path", type=str, required=True, help="Path to TSV dictionary file")
    parser.add_argument("--tsv_format", type=str, required=True, choices=["lz78", "compressed"],
                        help="TSV format: lz78 (code/parent/char/pattern) or compressed (node/edge/parent/code/pattern)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for tokenizer files")
    args = parser.parse_args()

    print(f"Building LZ78 tokenizer from {args.tsv_path} (format: {args.tsv_format})")

    # Build tokenizer
    tokenizer = LZ78Tokenizer.from_tsv(args.tsv_path, args.tsv_format)
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"BOS token ID: {tokenizer.get_bos_token_id()}")

    # Save
    tokenizer.save(args.output_dir)

    # Verify roundtrip
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Python is a great programming language.",
        "12345 + 67890 = 80235",
        "Special chars: @#$%^&*(){}[]|\\<>?/~`",
        "Unicode: café résumé naïve über",
        "Newlines:\nand\ttabs\there",
        "",
    ]
    verify_roundtrip(tokenizer, test_texts)

    # Test special tokens
    print(f"\nSpecial tokens: {tokenizer.get_special_tokens()}")
    bos_id = tokenizer.get_bos_token_id()
    ids_with_bos = tokenizer.encode("Hello", prepend="<|bos|>")
    print(f"Encode 'Hello' with BOS: {ids_with_bos[:5]}... (BOS={bos_id})")
    assert ids_with_bos[0] == bos_id, "BOS token not prepended correctly"

    # Test metadata
    metadata = tokenizer.get_token_metadata(mode="structured")
    print(f"\nStructured metadata shape: {metadata.shape}")
    print(f"Sample metadata (first 5 codes):")
    for i in range(1, min(6, metadata.shape[0])):
        print(f"  code {i}: parent={metadata[i, 0].item()}, char_byte={metadata[i, 1].item()} ({chr(metadata[i, 1].item()) if 32 <= metadata[i, 1].item() < 127 else '?'})")

    # Reload and verify
    print("\n--- Reload verification ---")
    tokenizer2 = LZ78Tokenizer.from_directory(args.output_dir)
    assert tokenizer2.get_vocab_size() == tokenizer.get_vocab_size()
    test_text = "The quick brown fox"
    ids1 = tokenizer.encode(test_text)
    ids2 = tokenizer2.encode(test_text)
    assert ids1 == ids2, f"Encode mismatch after reload: {ids1} != {ids2}"
    print("Reload verification passed!")

    print("\nDone!")


if __name__ == "__main__":
    main()
