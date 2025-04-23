import onnx

model_path = "/home/bruno/ESCI/TFG/parametric_UMAP_code/tests_comarison/Using_source_code_from_library/pumap_model_tester.onnx"
model = onnx.load(model_path)

print("Available model outputs:", [output.name for output in model.graph.output])
