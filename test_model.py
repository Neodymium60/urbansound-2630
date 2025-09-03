#!/usr/bin/env python3
"""
Diagnostic script to validate a TFLite model (e.g., YAMNet).

Prints:
- Input/output tensor shapes
- Dtype and quantization
- Flags any load/inference issues
"""

def test_tflite_model(model_path="/app/model.tflite"):
    try:
        # Try both common TFLite runtimes
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter

        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("✅ Model loaded:", model_path)
        print("\n===== Input Details =====")
        for t in input_details:
            print(f"Name:         {t['name']}")
            print(f"Shape:        {t['shape']}")
            print(f"Dtype:        {t['dtype']}")
            print(f"Quantization: {t['quantization']}")
            print("-" * 32)

        print("\n===== Output Details =====")
        for t in output_details:
            print(f"Name:         {t['name']}")
            print(f"Shape:        {t['shape']}")
            print(f"Dtype:        {t['dtype']}")
            print(f"Quantization: {t['quantization']}")
            print("-" * 32)

    except Exception as e:
        print(f"❌ Error inspecting model: {e}")

if __name__ == "__main__":
    test_tflite_model()

