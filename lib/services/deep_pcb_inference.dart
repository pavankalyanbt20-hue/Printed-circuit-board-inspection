
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:image/image.dart' as img;

import 'deep_pcb_preprocessor.dart';

class DeepPcbDetection {
  final String className;
  final int classId;
  final double confidence;
  final double x1;
  final double y1;
  final double x2;
  final double y2;
  final int imageWidth;
  final int imageHeight;

  const DeepPcbDetection({
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

class DeepPcbInferenceService {
  static const String inputName = 'images';
  static const String outputName = 'output0';

  static const List<int> inputShape = [1, 3, 640, 640];

  static const int predictionCount = 8400;
  static const int channelCount = 10;

  static const double confidenceThreshold = 0.25;
  static const double iouThreshold = 0.50;
  static const int maxDetections = 300;

  static const List<String> classNames = [
    'Open',
    'Short',
    'Mousebite',
    'Spur',
    'Spurious Copper',
    'Pin Hole',
  ];

  Future<List<DeepPcbDetection>> runInference({
    required OrtSession session,
    required File imageFile,
  }) async {
    OrtValue? inputTensor;
    Map<String, OrtValue>? outputs;

    try {
      if (!session.inputNames.contains(inputName)) {
        throw Exception('Model input "images" was not found.');
      }

      if (!session.outputNames.contains(outputName)) {
        throw Exception('Model output "output0" was not found.');
      }

      // 1. Read original image dimensions.
      final imageBytes = await imageFile.readAsBytes();
      final decodedImage = img.decodeImage(imageBytes);

      if (decodedImage == null) {
        throw Exception('Unable to decode original PCB image.');
      }

      final original = img.bakeOrientation(decodedImage);
      final originalWidth = original.width;
      final originalHeight = original.height;

      // 2. Preprocess image.
      final Float32List inputData =
          await DeepPcbPreprocessor.preprocess(imageFile);

      if (inputData.length != 3 * 640 * 640) {
        throw Exception('Invalid input tensor size.');
      }

      // 3. Export Android input tensor for validation.
      final tensorPath =
          await DeepPcbPreprocessor.exportTensor(inputData);

      debugPrint('Android input tensor exported: $tensorPath');
      debugPrint('Tensor values: ${inputData.length}');
      debugPrint('Tensor bytes: ${inputData.lengthInBytes}');

      // 4. Create ONNX input tensor.
      inputTensor = await OrtValue.fromList(
        inputData,
        inputShape,
      );

      // 5. Run inference.
      final stopwatch = Stopwatch()..start();

      outputs = await session.run({
        inputName: inputTensor,
      });

      stopwatch.stop();

      // 6. Retrieve model output.
      final outputTensor = outputs[outputName];

      if (outputTensor == null) {
        throw Exception('Model output0 was not returned.');
      }

      final outputData = await outputTensor.asFlattenedList();

      if (outputData.length != channelCount * predictionCount) {
        throw Exception(
          'Unexpected output length: ${outputData.length}',
        );
      }

      // 7. Export raw predictions to CSV.
      final buffer = StringBuffer();

      buffer.writeln(
        'prediction_index,box_x,box_y,box_w,box_h,'
        'open,short,mousebite,spur,spurious_copper,pin_hole',
      );

      for (int i = 0; i < predictionCount; i++) {
        final values = <String>[i.toString()];

        for (int channel = 0; channel < channelCount; channel++) {
          final value =
              (outputData[channel * predictionCount + i] as num)
                  .toDouble();

          values.add(value.toString());
        }

        buffer.writeln(values.join(','));
      }

      final csvFile = File(
        '${Directory.systemTemp.path}/deeppcb_android_output.csv',
      );

      await csvFile.writeAsString(
        buffer.toString(),
        flush: true,
      );

      // 8. Decode predictions and apply NMS.
      final detections = decodePredictions(
        outputData: outputData,
        originalWidth: originalWidth,
        originalHeight: originalHeight,
      );

      debugPrint('======================================');
      debugPrint('DEEPPCB INFERENCE COMPLETED');
      debugPrint('Input shape: $inputShape');
      debugPrint('Output shape: [1, 10, 8400]');
      debugPrint('Output values: ${outputData.length}');
      debugPrint(
        'Inference time: ${stopwatch.elapsedMilliseconds} ms',
      );
      debugPrint('CSV exported: ${csvFile.path}');
      debugPrint('Tensor exported: $tensorPath');
      debugPrint('Detections after NMS: ${detections.length}');

      for (final detection in detections) {
        debugPrint(
          '${detection.className}: '
          '${(detection.confidence * 100).toStringAsFixed(2)}%',
        );
      }

      debugPrint('======================================');

      return detections;
    } catch (error, stackTrace) {
      debugPrint('DEEPPCB INFERENCE ERROR: $error');
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

  List<DeepPcbDetection> decodePredictions({
    required List<dynamic> outputData,
    required int originalWidth,
    required int originalHeight,
  }) {
    final scale = math.min(
      640 / originalWidth,
      640 / originalHeight,
    );

    final resizedWidth = (originalWidth * scale).round();
    final resizedHeight = (originalHeight * scale).round();

    final left = (640 - resizedWidth) ~/ 2;
    final top = (640 - resizedHeight) ~/ 2;

    final candidates = <DeepPcbDetection>[];

    for (int i = 0; i < predictionCount; i++) {
      double bestScore = 0;
      int bestClass = -1;

      // Find the highest class score for this prediction.
      for (int c = 0; c < 6; c++) {
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

      // Convert center XYWH into corner coordinates.
      final inputX1 = centerX - boxWidth / 2;
      final inputY1 = centerY - boxHeight / 2;
      final inputX2 = centerX + boxWidth / 2;
      final inputY2 = centerY + boxHeight / 2;

      // Reverse letterbox padding and resizing.
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
        DeepPcbDetection(
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

    // Highest confidence predictions first.
    candidates.sort(
      (a, b) => b.confidence.compareTo(a.confidence),
    );

    // Class-aware Non-Maximum Suppression.
    final selected = <DeepPcbDetection>[];

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

  double calculateIoU(
    DeepPcbDetection a,
    DeepPcbDetection b,
  ) {
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

    if (unionArea <= 0) {
      return 0;
    }

    return intersectionArea / unionArea;
  }
}
