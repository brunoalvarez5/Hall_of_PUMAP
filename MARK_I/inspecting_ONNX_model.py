import onnx

model_path = ".model/pumap_model_tester.onnx"
model = onnx.load(model_path)

print("Available model outputs:", [output.name for output in model.graph.output])
