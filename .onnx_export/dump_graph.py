"""导出 nmp.onnx 的完整拓扑与权重信息（只读分析）。"""
import onnx
from onnx import numpy_helper

MODEL = r"d:\MusicRecorder\BasicPitch\nmp.onnx"

m = onnx.load(MODEL)
g = m.graph

print("=" * 80)
print("GRAPH")
print("=" * 80)
print("name:", g.name)
print("opset:", [(o.domain or "ai.onnx", o.version) for o in m.opset_import])
print("producer:", m.producer_name, m.producer_version)
print()

print("INPUTS:")
for i in g.input:
    dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in i.type.tensor_type.shape.dim]
    print("  ", i.name, i.type.tensor_type.elem_type, dims)

print("OUTPUTS:")
for o in g.output:
    dims = [d.dim_value if d.HasField("dim_value") else d.dim_param for d in o.type.tensor_type.shape.dim]
    print("  ", o.name, o.type.tensor_type.elem_type, dims)

print()
print("=" * 80)
print("INITIALIZERS (%d)" % len(g.initializer))
print("=" * 80)
total_params = 0
for init in g.initializer:
    arr = numpy_helper.to_array(init)
    total_params += arr.size
    print("  %-40s %-12s %s" % (init.name, str(arr.dtype), str(arr.shape)))
print("total params:", total_params)
print()

print("=" * 80)
print("NODES (%d)" % len(g.node))
print("=" * 80)
for idx, n in enumerate(g.node):
    ins = list(n.input)
    outs = list(n.output)
    attrs = []
    for a in n.attribute:
        if a.type == onnx.AttributeProto.INT:
            attrs.append("%s=%d" % (a.name, a.i))
        elif a.type == onnx.AttributeProto.FLOAT:
            attrs.append("%s=%g" % (a.name, a.f))
        elif a.type == onnx.AttributeProto.STRING:
            attrs.append("%s=%s" % (a.name, a.s.decode()))
        elif a.type == onnx.AttributeProto.INTS:
            attrs.append("%s=%s" % (a.name, list(a.ints)))
        elif a.type == onnx.AttributeProto.FLOATS:
            attrs.append("%s=%s" % (a.name, [round(x, 6) for x in a.floats]))
        elif a.type == onnx.AttributeProto.TENSOR:
            t = numpy_helper.to_array(a.t)
            attrs.append("%s=<tensor %s %s>" % (a.name, str(t.dtype), str(t.shape)))
        else:
            attrs.append("%s=<type %d>" % (a.name, a.type))
    print("[%3d] %-16s in=%s out=%s  {%s}" % (idx, n.op_type, ins, outs, ", ".join(attrs)))