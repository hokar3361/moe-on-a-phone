#!/usr/bin/env python3
# Rewrite a (split) GGUF so that the first file holds every non-expert tensor and the
# following files hold only MoE expert tensors ("_exps."). The patched loader does not
# map expert-only files (LLAMA_UNMAPPED_EXPERT_FILES=1), which keeps the mapped address
# space small enough for iOS. Tensor bytes are copied unchanged.
# usage: split_experts.py <gguf-py dir> <out prefix> <max expert file GiB> <first split file> [more split files...]
import sys
sys.path.insert(0, sys.argv[1])
import gguf

out_prefix = sys.argv[2]
max_bytes = int(float(sys.argv[3]) * (1 << 30))
readers = [gguf.GGUFReader(p) for p in sys.argv[4:]]

tensors = [t for r in readers for t in r.tensors]
dense = [t for t in tensors if "_exps." not in t.name]
experts = [t for t in tensors if "_exps." in t.name]

groups = [dense]
cur, size = [], 0
for t in experts:
    if cur and size + t.n_bytes > max_bytes:
        groups.append(cur)
        cur, size = [], 0
    cur.append(t)
    size += t.n_bytes
if cur:
    groups.append(cur)

n = len(groups)
arch = readers[0].fields[gguf.Keys.General.ARCHITECTURE].contents()
for i, group in enumerate(groups):
    path = f"{out_prefix}-{i + 1:05d}-of-{n:05d}.gguf"
    w = gguf.GGUFWriter(path, arch)
    if i == 0:
        for field in readers[0].fields.values():
            if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF.") or field.name.startswith("split."):
                continue
            vt = field.types[0]
            sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
            w.add_key_value(field.name, field.contents(), vt, sub_type=sub)
    w.add_uint16("split.no", i)
    w.add_uint16("split.count", n)
    w.add_int32("split.tensors.count", len(tensors))
    for t in group:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in group:
        w.write_tensor_data(t.data, tensor_endianess=readers[0].endianess)
    w.close()
    print(f"{path}: {len(group)} tensors, {sum(t.n_bytes for t in group) / (1 << 30):.2f} GiB", flush=True)
