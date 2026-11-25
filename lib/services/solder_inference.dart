
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;

import 'deep_pcb_preprocessor.dart';

class SolderDetection {
  final String className;
  final int classId;
  final double confidence;
  final double x1;
  final double y1;
  final double x2;
  final double y2;
  final int imageWidth;
  final int imageHeight;

  const SolderDetection({
    required this.className,
    required this.classId,
    required this.confidence,
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    required this.imageWidth,
    required this.imageHeight,
  });
}

class SolderInferenceService {
  static const String inputName = 'images';
  static const String outputName = 'output0';

  static const int inputSize = 640;
  static const int predictionCount = 8400;
  static const int channelCount = 13;

  static const double confidenceThreshold = 0.25;
  static const double iouThreshold = 0.50;
  static const int maxDetections = 300;

  static const List<String> classNames = [
    'Insufficient Solder',
    'Excess Solder',
    'Solder Bridge',
    'Solder Void',
    'Missing Component',
    'Shifted Component',
    'Burn',
    'Physical Damage',
    'Oxidation',
  ];

  Future<List<SolderDetection>> runInference({
    required OrtSession session,
    required File imageFile,
  }) async {
    OrtValue? inputTensor;
    Map<String, OrtValue>? outputs;

    try {
      if (!session.inputNames.contains(inputName)) {
        throw Exception('Solder model input "images" not found.');
      }

      if (!session.outputNames.contains(outputName)) {
        throw Exception('Solder model output "output0" not found.');
      }

      // Read original image dimensions after orientation correction.
      final inputData = await DeepPcbPreprocessor.preprocess(imageFile);

      // Obtain original dimensions for mapping boxes.
      // The same image is used by both models.
      final bytes = await imageFile.readAsBytes();

      // Decode using the shared image library.
      // Import image package is required for dimensions.
      final decoded = _decodeOriginalDimensions(bytes);

      final originalWidth = decoded.width;
      final originalHeight = decoded.height;

      if (inputData.length != 3 * inputSize * inputSize) {
        throw Exception('Invalid solder input tensor size.');
      }

      inputTensor = await OrtValue.fromList(
        inputData,
        [1, 3, inputSize, inputSize],
      );

      final stopwatch = Stopwatch()..start();

      outputs = await session.run({
        inputName: inputTensor,
      });

      stopwatch.stop();

      final outputTensor = outputs[outputName];

      if (outputTensor == null) {
        throw Exception('Solder output0 was not returned.');
      }

      final outputData = await outputTensor.asFlattenedList();

      if (outputData.length != channelCount * predictionCount) {
        throw Exception(
          'Unexpected solder output length: ${outputData.length}',
        );
      }

      final detections = decodePredictions(
        outputData: outputData,
        originalWidth: originalWidth,
        originalHeight: originalHeight,
      );

      debugPrint('SOLDER INFERENCE COMPLETED');
      debugPrint('Output shape: [1, 13, 8400]');
      debugPrint('Inference time: ${stopwatch.elapsedMilliseconds} ms');
      debugPrint('Solder detections: ${detections.length}');

      return detections;
    } catch (error, stackTrace) {
      debugPrint('SOLDER INFERENCE ERROR: $error');
      debugPrint('$stackTrace');
      rethrow;
    } finally {
      if (inputTensor != null) {
        await inputTensor.dispose();
      }

      if (outputs != null) {
        for (final tensor in outputs.values) {
          await tensor.dispose();
        }
      }
    }
  }

  // Decode the output in channel-first format [1, 13, 8400].
  List<SolderDetection> decodePredictions({
    required List<dynamic> outputData,
    required int originalWidth,
    required int originalHeight,
  }) {
    final scale = math.min(
      inputSize / originalWidth,
      inputSize / originalHeight,
    );

    final resizedWidth = (originalWidth * scale).round();
    final resizedHeight = (originalHeight * scale).round();

    final left = (inputSize - resizedWidth) ~/ 2;
    final top = (inputSize - resizedHeight) ~/ 2;

    final candidates = <SolderDetection>[];

    for (int i = 0; i < predictionCount; i++) {
      double bestScore = 0;
      int bestClass = -1;

      for (int c = 0; c < classNames.length; c++) {
        final score =
            (outputData[(4 + c) * predictionCount + i] as num)
                .toDouble();

        if (score > bestScore) {
          bestScore = score;
          bestClass = c;
        }
      }

      if (bestScore < confidenceThreshold || bestClass < 0) {
        continue;
      }

      final centerX =
          (outputData[i] as num).toDouble();
      final centerY =
          (outputData[predictionCount + i] as num).toDouble();
      final boxWidth =
          (outputData[2 * predictionCount + i] as num).toDouble();
      final boxHeight =
          (outputData[3 * predictionCount + i] as num).toDouble();

      final inputX1 = centerX - boxWidth / 2;
      final inputY1 = centerY - boxHeight / 2;
      final inputX2 = centerX + boxWidth / 2;
      final inputY2 = centerY + boxHeight / 2;

      final x1 = ((inputX1 - left) / scale)
          .clamp(0.0, originalWidth.toDouble());
      final y1 = ((inputY1 - top) / scale)
          .clamp(0.0, originalHeight.toDouble());
      final x2 = ((inputX2 - left) / scale)
          .clamp(0.0, originalWidth.toDouble());
      final y2 = ((inputY2 - top) / scale)
          .clamp(0.0, originalHeight.toDouble());

      if (x2 <= x1 || y2 <= y1) {
        continue;
      }

      candidates.add(
        SolderDetection(
          className: classNames[bestClass],
          classId: bestClass,
          confidence: bestScore,
          x1: x1,
          y1: y1,
          x2: x2,
          y2: y2,
          imageWidth: originalWidth,
          imageHeight: originalHeight,
        ),
      );
    }

    candidates.sort(
      (a, b) => b.confidence.compareTo(a.confidence),
    );

    final selected = <SolderDetection>[];

    for (final candidate in candidates) {
      bool shouldKeep = true;

      for (final existing in selected) {
        if (candidate.classId == existing.classId &&
            calculateIoU(candidate, existing) > iouThreshold) {
          shouldKeep = false;
          break;
        }
      }

      if (shouldKeep) {
        selected.add(candidate);
      }

      if (selected.length >= maxDetections) {
        break;
      }
    }

    return selected;
  }

  double calculateIoU(SolderDetection a, SolderDetection b) {
    final intersectionX1 = math.max(a.x1, b.x1);
    final intersectionY1 = math.max(a.y1, b.y1);
    final intersectionX2 = math.min(a.x2, b.x2);
    final intersectionY2 = math.min(a.y2, b.y2);

    final intersectionWidth =
        math.max(0.0, intersectionX2 - intersectionX1);
    final intersectionHeight =
        math.max(0.0, intersectionY2 - intersectionY1);

    final intersectionArea =
        intersectionWidth * intersectionHeight;

    final areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    final areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

    final unionArea = areaA + areaB - intersectionArea;

    if (unionArea <= 0) return 0;

    return intersectionArea / unionArea;
  }

  
  // Returns orientation-corrected dimensions of the original image.
  _ImageDimensions _decodeOriginalDimensions(List<int> bytes) {
    final decodedImage = img.decodeImage(
      Uint8List.fromList(bytes),
    );

    if (decodedImage == null) {
      throw Exception('Unable to decode original solder image.');
    }

    final orientedImage = img.bakeOrientation(decodedImage);

    return _ImageDimensions(
      orientedImage.width,
      orientedImage.height,
    );
  }
}

// Stores the original image dimensions.
class _ImageDimensions {
  final int width;
  final int height;

  const _ImageDimensions(this.width, this.height);
}
