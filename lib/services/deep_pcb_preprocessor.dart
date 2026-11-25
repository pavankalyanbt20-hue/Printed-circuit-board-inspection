
import 'dart:io';
import 'dart:typed_data';

import 'package:image/image.dart' as img;

class DeepPcbPreprocessor {
  static const int inputSize = 640;
  static const int paddingValue = 114;

  static Future<Float32List> preprocess(File imageFile) async {
    // Read original image.
    final bytes = await imageFile.readAsBytes();

    final decodedImage = img.decodeImage(bytes);

    if (decodedImage == null) {
      throw Exception('Unable to decode PCB image.');
    }

    // Apply image orientation.
    final original = img.bakeOrientation(decodedImage);

    final originalWidth = original.width;
    final originalHeight = original.height;

    // Calculate aspect-preserving resize ratio.
    final scale = [
      inputSize / originalWidth,
      inputSize / originalHeight,
    ].reduce((a, b) => a < b ? a : b);

    final resizedWidth =
        (originalWidth * scale).round().clamp(1, inputSize);

    final resizedHeight =
        (originalHeight * scale).round().clamp(1, inputSize);

    // Resize using linear interpolation.
    final resized = img.copyResize(
      original,
      width: resizedWidth,
      height: resizedHeight,
      interpolation: img.Interpolation.linear,
    );

    // Calculate centered padding.
    final left = (inputSize - resizedWidth) ~/ 2;
    final top = (inputSize - resizedHeight) ~/ 2;

    // Create a 640 x 640 RGB image with padding value 114.
    final canvas = img.Image(
      width: inputSize,
      height: inputSize,
      numChannels: 3,
    );

    img.fill(
      canvas,
      color: img.ColorRgb8(
        paddingValue,
        paddingValue,
        paddingValue,
      ),
    );

    img.compositeImage(
      canvas,
      resized,
      dstX: left,
      dstY: top,
    );

    // Prepare channel-first RGB Float32 input.
    const planeSize = inputSize * inputSize;

    final input = Float32List(3 * planeSize);

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = canvas.getPixel(x, y);

        final index = y * inputSize + x;

        input[index] = pixel.r / 255.0;
        input[planeSize + index] = pixel.g / 255.0;
        input[2 * planeSize + index] = pixel.b / 255.0;
      }
    }

    return input;
  }

  // Export the exact preprocessing tensor for validation.
  static Future<String> exportTensor(Float32List input) async {
    if (input.length != 3 * inputSize * inputSize) {
      throw Exception('Invalid input tensor length.');
    }

    final directory = Directory.systemTemp.path;
    final file = File('$directory/android_input_tensor.bin');

    final bytes = input.buffer.asUint8List(
      input.offsetInBytes,
      input.lengthInBytes,
    );

    await file.writeAsBytes(bytes, flush: true);

    return file.path;
  }
}
