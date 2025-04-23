import onnx
import onnxruntime as ort

# Load the ONNX model
model_path = "/home/bruno/ESCI/TFG/parametric_UMAP_code/MARK_III/Mark_III_model.onnx"
model = onnx.load(model_path)

# Check input dimensions
input_all = model.graph.input
input_dims = input_all[0].type.tensor_type.shape.dim

print("Input shape:")
for dim in input_dims:
    dim_val = dim.dim_value if dim.HasField("dim_value") else "dynamic"
    print(f" - {dim_val}")
