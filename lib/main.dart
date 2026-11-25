import 'dart:io';
import 'dart:convert';
import 'dart:math' as math;

import 'package:material_ui/material_ui.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:path_provider/path_provider.dart';

import 'services/deep_pcb_inference.dart';
import 'services/severity_assessment.dart';

void main() {
  runApp(const DefectDetectionApp());
}

class DefectDetectionApp extends StatelessWidget {
  const DefectDetectionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'DeepPCB Defect Detection',
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF087E8B),
        ),
        scaffoldBackgroundColor: const Color(0xFFF4F7FA),
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  static const MethodChannel _llamaChannel =
      MethodChannel('industrial_defect_detection/llama');

  static const int expectedModelSize = 802247040;
  static const int expectedTokenizerSize = 9085657;

  final ImagePicker picker = ImagePicker();
  final OnnxRuntime onnxRuntime = OnnxRuntime();

  final DeepPcbInferenceService inferenceService =
      DeepPcbInferenceService();

  final SeverityAssessmentService severityAssessmentService =
      SeverityAssessmentService();

  List<Map<String, dynamic>> inspectionHistory = [];
  bool isSavingHistory = false;

  File? selectedImage;
  OrtSession? modelSession;

  List<DeepPcbDetection> detections = [];
  List<SeverityResult> severityResults = [];

  bool isPickingImage = false;
  bool isLoadingModel = false;
  bool isModelLoaded = false;
  bool isRunningInference = false;
  bool hasInspectionResult = false;

  bool isPreparingLlamaFiles = false;
  bool llamaFilesVerified = false;
  bool isLoadingLlama = false;
  bool isLlamaLoaded = false;
  bool isGeneratingLlama = false;

  String llamaFileStatus = 'Checking bundled Llama files...';
  String llamaStatus = 'Llama model not loaded';
  String llamaResponse = '';

  String? llamaModelPath;
  String? llamaTokenizerPath;

  @override
  void initState() {
    super.initState();

    WidgetsBinding.instance.addPostFrameCallback((_) {
      prepareLlamaFiles();
      loadInspectionHistory();
    });
  }

  // ------------------------------------------------------------
  // DEEPPCB MODEL LOADING
  // ------------------------------------------------------------

  Future<void> loadDeepPcbModel() async {
    if (isLoadingModel || isModelLoaded) return;

    setState(() {
      isLoadingModel = true;
    });

    OrtSession? session;

    try {
      debugPrint('Loading DeepPCB ONNX model...');

      session = await onnxRuntime.createSessionFromAsset(
        'assets/models/deeppcb_best.onnx',
      );

      if (!mounted) {
        await session.close();
        return;
      }

      modelSession = session;

      setState(() {
        isModelLoaded = true;
      });

      debugPrint('DEEPPCB MODEL LOADED SUCCESSFULLY');
      debugPrint('Input names: ${session.inputNames}');
      debugPrint('Output names: ${session.outputNames}');

      showMessage('DeepPCB model loaded successfully.');
    } catch (e, stackTrace) {
      debugPrint('DEEPPCB MODEL LOADING ERROR: $e');
      debugPrint('$stackTrace');

      if (session != null) {
        await session.close();
      }

      if (mounted) {
        showMessage('Unable to load DeepPCB model: $e');
      }
    } finally {
      if (mounted) {
        setState(() {
          isLoadingModel = false;
        });
      }
    }
  }

  // ------------------------------------------------------------
  // DEEPPCB INFERENCE
  // ------------------------------------------------------------

  Future<void> runDeepPcbInspection() async {
    if (isRunningInference) return;

    if (modelSession == null || !isModelLoaded) {
      showMessage('Please load the DeepPCB model first.');
      return;
    }

    if (selectedImage == null) {
      showMessage('Please select a PCB image first.');
      return;
    }

    setState(() {
      isRunningInference = true;
      hasInspectionResult = false;
      detections = [];
      severityResults = [];
      llamaResponse = '';
    });

    try {
      final imageForRecord = selectedImage!;

      final results = await inferenceService.runInference(
        session: modelSession!,
        imageFile: imageForRecord,
      );

      final assessedResults = results.map((detection) {
        return severityAssessmentService.assess(
          defectName: detection.className,
          confidence: detection.confidence,
          x: (detection.x1 + detection.x2) / 2,
          y: (detection.y1 + detection.y2) / 2,
        );
      }).toList();

      if (!mounted) return;

      setState(() {
        detections = results;
        severityResults = assessedResults;
        hasInspectionResult = true;
      });

      String report;
      String aiStatus;

      if (results.isEmpty) {
        report =
            'No DeepPCB defects were detected above the model confidence threshold. '
            'This does not prove that the PCB is defect-free. '
            'No defect-specific recommendations have been generated.';

        aiStatus =
            'No defects detected; AI defect explanation not required.';
      } else if (isLlamaLoaded) {
        setState(() {
          llamaStatus =
              'Generating report from actual inspection results...';
        });

        try {
          final prompt = buildInspectionPrompt(
            results,
            assessedResults,
          );

          report = await generateLlamaResponse(prompt);
          aiStatus = 'AI report generated';
        } catch (e) {
          report = 'AI report could not be generated: $e';
          aiStatus = 'AI generation failed';
        }
      } else {
        report =
            'Llama model is not loaded. Load Llama 3.2 1B and run '
            'the inspection again to generate an AI explanation. '
            'The detection and severity results are saved.';

        aiStatus = 'Llama model not loaded';
      }

      if (mounted) {
        setState(() {
          llamaResponse = report;
          llamaStatus = aiStatus;
        });
      }

      await saveInspectionRecord(
        imageForRecord,
        results,
        assessedResults,
        report,
        aiStatus,
      );

      if (mounted) {
        showMessage(
          results.isEmpty
              ? 'Inspection complete. No defects above threshold.'
              : 'Inspection complete: ${results.length} defect(s). '
                  'AI report and history updated.',
        );
      }
    } catch (e, stackTrace) {
      debugPrint('DEEPPCB INSPECTION ERROR: $e');
      debugPrint('$stackTrace');

      if (mounted) {
        showMessage('Inspection failed: $e');
      }
    } finally {
      if (mounted) {
        setState(() {
          isRunningInference = false;
        });
      }
    }
  }

  // ------------------------------------------------------------
  // SHORT LLAMA PROMPT
  // ------------------------------------------------------------

  String buildInspectionPrompt(
  List<DeepPcbDetection> found,
  List<SeverityResult> assessed,
) {
  const classNames = <String, String>{
    'open': 'Open Circuit',
    'open_circuit': 'Open Circuit',
    'short': 'Short Circuit',
    'short_circuit': 'Short Circuit',
    'mousebite': 'Mousebite',
    'mouse_bite': 'Mousebite',
    'spur': 'Spur',
    'spurious_copper': 'Spurious Copper',
    'missing_hole': 'Missing Hole',
    'pin_hole': 'Missing Hole',
  };

  const descriptions = <String, String>{
    'Open Circuit':
        'Open Circuit is a break in a conductive path that stops current flow.',
    'Short Circuit':
        'Short Circuit to prevent overheating, component damage, or fire.',
    'Mousebite':
        'Mouse bites are tiny, closely spaced non-plated drill holes.',
    'Spurious Copper':
        'Spurious copper is unwanted copper left on a PCB due to incomplete etching.',
    'Missing Hole':
        'A tiny hole reaching down to the base copper.',
    'Spur':
        'Spur is a protrusion of copper sticking out from the edge of a track.',
  };

  final detectedDefects = <String>[];

  for (final detection in found) {
    final className = detection.className
        .toLowerCase()
        .trim()
        .replaceAll('-', '_')
        .replaceAll(' ', '_');

    final defect = classNames[className];

    if (defect != null && !detectedDefects.contains(defect)) {
      detectedDefects.add(defect);
    }
  }

  final responseText = detectedDefects
      .map((defect) => descriptions[defect]!)
      .join(' ');

  return '<|begin_of_text|>'
      '<|start_header_id|>system<|end_header_id|>\n\n'
      'Use only the following detected defect descriptions. '
      'Return them exactly as provided, in one continuous paragraph. '
      'Do not add, remove, or modify any words. '
      'Do not include headings, numbering, or extra text.\n\n'
      '$responseText'
      '<|eot_id|>'
      '<|start_header_id|>user<|end_header_id|>\n\n'
      'Return the provided descriptions exactly as one paragraph.'
      '<|eot_id|>'
      '<|start_header_id|>assistant<|end_header_id|>\n\n';
}
  // ------------------------------------------------------------
  // INSPECTION HISTORY
  // ------------------------------------------------------------

  Future<File> _historyFile() async {
    final dir = await getApplicationDocumentsDirectory();

    return File(
      '${dir.path}${Platform.pathSeparator}inspection_history.json',
    );
  }

  Future<void> loadInspectionHistory() async {
    try {
      final file = await _historyFile();

      if (!await file.exists()) return;

      final decoded = jsonDecode(await file.readAsString());

      if (decoded is List && mounted) {
        setState(() {
          inspectionHistory = decoded
              .whereType<Map>()
              .map((e) => Map<String, dynamic>.from(e))
              .toList();
        });
      }
    } catch (e) {
      debugPrint('History load error: $e');
    }
  }

  Future<void> saveInspectionRecord(
    File image,
    List<DeepPcbDetection> found,
    List<SeverityResult> assessed,
    String aiReport,
    String aiStatus,
  ) async {
    if (isSavingHistory) return;

    isSavingHistory = true;

    try {
      final dir = await getApplicationDocumentsDirectory();

      final imageDir = Directory(
        '${dir.path}${Platform.pathSeparator}inspection_images',
      );

      await imageDir.create(recursive: true);

      final stamp = DateTime.now().microsecondsSinceEpoch;

      final ext = image.path.contains('.')
          ? image.path.substring(image.path.lastIndexOf('.'))
          : '.jpg';

      final savedImage = await image.copy(
        '${imageDir.path}${Platform.pathSeparator}inspection_$stamp$ext',
      );

      final overall = found.isEmpty
          ? 'Not applicable'
          : severityLabel(
              severityAssessmentService.overallSeverity(assessed),
            );

      final record = <String, dynamic>{
        'id': '$stamp',
        'inspectedAt': DateTime.now().toIso8601String(),
        'imagePath': savedImage.path,
        'totalDefects': found.length,
        'overallSeverity': overall,
        'aiReport': aiReport,
        'aiStatus': aiStatus,
        'detections': List.generate(found.length, (i) {
          return {
            'className': found[i].className,
            'classId': found[i].classId,
            'confidence': found[i].confidence,
            'x1': found[i].x1,
            'y1': found[i].y1,
            'x2': found[i].x2,
            'y2': found[i].y2,
            'severity': assessed[i].severityLabel,
            'status': assessed[i].status,
          };
        }),
      };

      final updated = [record, ...inspectionHistory];

      final file = await _historyFile();

      await file.writeAsString(
        jsonEncode(updated),
        flush: true,
      );

      if (mounted) {
        setState(() {
          inspectionHistory = updated;
        });
      }
    } catch (e) {
      debugPrint('History save error: $e');

      if (mounted) {
        showMessage(
          'Inspection completed, but history could not be saved: $e',
        );
      }
    } finally {
      isSavingHistory = false;
    }
  }

  void openHistory() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => InspectionHistoryScreen(
          records: inspectionHistory,
        ),
      ),
    );
  }
    // ------------------------------------------------------------
  // AUTOMATIC LLAMA ASSET VERIFICATION
  // ------------------------------------------------------------

  Future<void> prepareLlamaFiles() async {
    if (isPreparingLlamaFiles || llamaFilesVerified) return;

    if (mounted) {
      setState(() {
        isPreparingLlamaFiles = true;
        llamaFileStatus = 'Preparing bundled Llama files...';
      });
    }

    try {
      final result =
          await _llamaChannel.invokeMapMethod<String, dynamic>(
        'prepareLlamaFiles',
      );

      if (result == null || result['success'] != true) {
        throw Exception(
          result?['message'] ??
              'Android could not verify Llama files.',
        );
      }

      final modelPath = result['modelPath'] as String?;
      final tokenizerPath = result['tokenizerPath'] as String?;

      final modelSize =
          (result['modelSize'] as num?)?.toInt();

      final tokenizerSize =
          (result['tokenizerSize'] as num?)?.toInt();

      if (modelPath == null || tokenizerPath == null) {
        throw Exception(
          'Android did not return both Llama file paths.',
        );
      }

      if (modelSize != expectedModelSize) {
        throw Exception(
          'Unexpected model size: $modelSize bytes. '
          'Expected $expectedModelSize bytes.',
        );
      }

      if (tokenizerSize != expectedTokenizerSize) {
        throw Exception(
          'Unexpected tokenizer size: $tokenizerSize bytes. '
          'Expected $expectedTokenizerSize bytes.',
        );
      }

      if (!mounted) return;

      setState(() {
        llamaModelPath = modelPath;
        llamaTokenizerPath = tokenizerPath;
        llamaFilesVerified = true;
        llamaFileStatus = 'Llama Files Verified';
        isLlamaLoaded = false;
        llamaStatus = 'Files verified. Ready to load Llama.';
        llamaResponse = '';
      });

      debugPrint('LLAMA MODEL FILE: $modelPath');
      debugPrint('MODEL SIZE: $modelSize bytes');
      debugPrint('LLAMA TOKENIZER FILE: $tokenizerPath');
      debugPrint('TOKENIZER SIZE: $tokenizerSize bytes');
    } on PlatformException catch (e) {
      debugPrint('LLAMA ASSET ERROR: ${e.code} ${e.message}');

      if (mounted) {
        setState(() {
          llamaFilesVerified = false;
          llamaFileStatus = 'Llama file verification failed.';
        });

        showMessage('Llama asset error: ${e.message}');
      }
    } catch (e, stackTrace) {
      debugPrint('LLAMA FILE VERIFICATION ERROR: $e');
      debugPrint('$stackTrace');

      if (mounted) {
        setState(() {
          llamaFilesVerified = false;
          llamaFileStatus = 'Llama file verification failed.';
        });

        showMessage('Llama file error: $e');
      }
    } finally {
      if (mounted) {
        setState(() {
          isPreparingLlamaFiles = false;
        });
      }
    }
  }

  // ------------------------------------------------------------
  // LLAMA MODEL LOADING
  // ------------------------------------------------------------

  Future<void> loadLlamaModel() async {
    if (isLoadingLlama || isLlamaLoaded) return;

    if (!llamaFilesVerified) {
      await prepareLlamaFiles();

      if (!llamaFilesVerified) {
        showMessage('Llama files are not verified yet.');
        return;
      }
    }

    setState(() {
      isLoadingLlama = true;
      llamaStatus = 'Loading Llama 3.2 1B model...';
      llamaResponse = '';
    });

    try {
      final result =
          await _llamaChannel.invokeMapMethod<String, dynamic>(
        'loadLlamaModel',
      );

      if (result == null || result['success'] != true) {
        throw Exception('Llama model loading failed.');
      }

      if (!mounted) return;

      setState(() {
        isLlamaLoaded = true;
        llamaStatus = 'Llama model loaded successfully.';
      });

      showMessage('Llama model loaded successfully.');
    } on PlatformException catch (e) {
      debugPrint('LLAMA LOAD ERROR: ${e.code} ${e.message}');

      if (mounted) {
        setState(() {
          llamaStatus = 'Llama loading failed: ${e.message}';
        });

        showMessage('Llama loading error: ${e.message}');
      }
    } catch (e, stackTrace) {
      debugPrint('LLAMA LOAD ERROR: $e');
      debugPrint('$stackTrace');

      if (mounted) {
        setState(() {
          llamaStatus = 'Llama loading failed: $e';
        });

        showMessage('Unable to load Llama: $e');
      }
    } finally {
      if (mounted) {
        setState(() {
          isLoadingLlama = false;
        });
      }
    }
  }

  // ------------------------------------------------------------
  // LLAMA TEXT GENERATION
  // ------------------------------------------------------------

  Future<String> generateLlamaResponse(String prompt) async {
    if (isGeneratingLlama) {
      throw Exception('Llama is already generating a response.');
    }

    if (!isLlamaLoaded) {
      throw Exception('Please load the Llama model first.');
    }

    setState(() {
      isGeneratingLlama = true;
      llamaResponse = '';
    });

    try {
      final result =
          await _llamaChannel.invokeMapMethod<String, dynamic>(
        'generateLlamaText',
        {
          'prompt': prompt,
        },
      );

      if (result == null || result['success'] != true) {
        throw Exception('Llama generation failed.');
      }

      final response = result['response'] as String?;

      if (response == null || response.trim().isEmpty) {
        throw Exception('Llama returned an empty response.');
      }

      debugPrint('LLAMA RESPONSE: $response');

      return response.trim();
    } on PlatformException catch (e) {
      debugPrint('LLAMA GENERATION ERROR: ${e.code} ${e.message}');

      throw Exception('Llama generation error: ${e.message}');
    } catch (e, stackTrace) {
      debugPrint('LLAMA GENERATION ERROR: $e');
      debugPrint('$stackTrace');

      rethrow;
    } finally {
      if (mounted) {
        setState(() {
          isGeneratingLlama = false;
        });
      }
    }
  }

  // ------------------------------------------------------------
  // IMAGE SELECTION
  // ------------------------------------------------------------

  Future<void> pickImage(ImageSource source) async {
    if (isPickingImage || isRunningInference) return;

    setState(() {
      isPickingImage = true;
    });

    try {
      final XFile? image = await picker.pickImage(
        source: source,
        imageQuality: 90,
        maxWidth: 2000,
        maxHeight: 2000,
      );

      if (image != null && mounted) {
        setState(() {
          selectedImage = File(image.path);
          detections = [];
          severityResults = [];
          hasInspectionResult = false;
        });
      }
    } catch (e) {
      if (mounted) {
        showMessage('Unable to select image: $e');
      }
    } finally {
      if (mounted) {
        setState(() {
          isPickingImage = false;
        });
      }
    }
  }

  void showImageOptions() {
    if (isRunningInference) return;

    showModalBottomSheet(
      context: context,
      builder: (context) {
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text(
                  'Select PCB Image',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 20),
                ListTile(
                  leading: const Icon(Icons.photo_library),
                  title: const Text('Choose from Gallery'),
                  onTap: () {
                    Navigator.pop(context);
                    pickImage(ImageSource.gallery);
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.camera_alt),
                  title: const Text('Capture using Camera'),
                  onTap: () {
                    Navigator.pop(context);
                    pickImage(ImageSource.camera);
                  },
                ),
                const SizedBox(height: 10),
              ],
            ),
          ),
        );
      },
    );
  }

  void removeImage() {
    if (isRunningInference) return;

    setState(() {
      selectedImage = null;
      detections = [];
      severityResults = [];
      hasInspectionResult = false;
    });
  }

  void showMessage(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        duration: const Duration(seconds: 5),
      ),
    );
  }

  @override
  void dispose() {
    final session = modelSession;

    if (session != null) {
      session.close();
    }

    super.dispose();
  }

  // ------------------------------------------------------------
  // MAIN USER INTERFACE
  // ------------------------------------------------------------

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: const Color(0xFFF4F7FA),
        title: const Text(
          'DeepPCB Inspector',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        actions: [
          IconButton(
            tooltip: 'Inspection history',
            onPressed: openHistory,
            icon: const Icon(Icons.history),
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(18),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(22),
                decoration: BoxDecoration(
                  color: const Color(0xFF102A43),
                  borderRadius: BorderRadius.circular(22),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Icon(
                      Icons.memory,
                      color: Color(0xFF48D6C8),
                      size: 38,
                    ),
                    const SizedBox(height: 18),
                    const Text(
                      'DeepPCB Defect Detection',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 23,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 9),
                    const Text(
                      'Offline AI-powered PCB defect inspection.',
                      style: TextStyle(
                        color: Colors.white70,
                        fontSize: 14,
                      ),
                    ),
                    const SizedBox(height: 20),
                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton.icon(
                        onPressed:
                            isPickingImage || isRunningInference
                                ? null
                                : showImageOptions,
                        icon: const Icon(Icons.add_photo_alternate),
                        label: Text(
                          isPickingImage
                              ? 'Opening...'
                              : 'Select PCB Image',
                        ),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: const Color(0xFF48D6C8),
                          foregroundColor: const Color(0xFF102A43),
                          padding: const EdgeInsets.all(15),
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 20),

              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: isLoadingModel || isModelLoaded
                      ? null
                      : loadDeepPcbModel,
                  icon: isLoadingModel
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                          ),
                        )
                      : Icon(
                          isModelLoaded
                              ? Icons.check_circle
                              : Icons.memory,
                        ),
                  label: Text(
                    isLoadingModel
                        ? 'Loading DeepPCB Model...'
                        : isModelLoaded
                            ? 'DeepPCB Model Loaded'
                            : 'Load DeepPCB Model',
                  ),
                ),
              ),

              const SizedBox(height: 12),

              SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: selectedImage != null &&
                          isModelLoaded &&
                          !isRunningInference
                      ? runDeepPcbInspection
                      : null,
                  icon: isRunningInference
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      : const Icon(Icons.search),
                  label: Text(
                    isRunningInference
                        ? 'Running DeepPCB Inspection...'
                        : 'Inspect PCB',
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF087E8B),
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.all(16),
                  ),
                ),
              ),

              const SizedBox(height: 24),

              const Text(
                'Llama AI Assistant',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),

              const SizedBox(height: 10),
                            Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Llama 3.2 1B Instruct',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 16,
                      ),
                    ),
                    const SizedBox(height: 6),
                    const Text(
                      'The model and tokenizer are bundled inside this '
                      'application and are verified automatically.',
                    ),
                    const SizedBox(height: 12),
                    Row(
                      children: [
                        if (isPreparingLlamaFiles)
                          const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                            ),
                          )
                        else
                          Icon(
                            llamaFilesVerified
                                ? Icons.check_circle
                                : Icons.error_outline,
                            color: llamaFilesVerified
                                ? Colors.green
                                : Colors.orange,
                          ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: Text(
                            isPreparingLlamaFiles
                                ? 'Verifying bundled Llama files...'
                                : llamaFileStatus,
                            style: TextStyle(
                              color: llamaFilesVerified
                                  ? Colors.green
                                  : Colors.blueGrey,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ],
                    ),
                    if (llamaFilesVerified) ...[
                      const SizedBox(height: 8),
                      const Text(
                        'Model: 802,247,040 bytes',
                        style: TextStyle(fontSize: 12),
                      ),
                      const Text(
                        'Tokenizer: 9,085,657 bytes',
                        style: TextStyle(fontSize: 12),
                      ),
                    ],
                    if (!llamaFilesVerified &&
                        !isPreparingLlamaFiles) ...[
                      const SizedBox(height: 10),
                      Align(
                        alignment: Alignment.centerLeft,
                        child: TextButton.icon(
                          onPressed: prepareLlamaFiles,
                          icon: const Icon(Icons.refresh),
                          label: const Text('Retry verification'),
                        ),
                      ),
                    ],
                  ],
                ),
              ),

              const SizedBox(height: 14),

              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Llama Runtime',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),

                    const SizedBox(height: 12),

                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton.icon(
                        onPressed: !llamaFilesVerified ||
                                isPreparingLlamaFiles ||
                                isLoadingLlama ||
                                isLlamaLoaded
                            ? null
                            : loadLlamaModel,
                        icon: isLoadingLlama
                            ? const SizedBox(
                                width: 18,
                                height: 18,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                ),
                              )
                            : Icon(
                                isLlamaLoaded
                                    ? Icons.check_circle
                                    : Icons.memory,
                              ),
                        label: Text(
                          isLoadingLlama
                              ? 'Loading Llama Model...'
                              : isLlamaLoaded
                                  ? 'Llama Model Loaded'
                                  : 'Load Llama Model',
                        ),
                      ),
                    ),

                    const SizedBox(height: 10),

                    Text(
                      llamaStatus,
                      style: TextStyle(
                        color: isLlamaLoaded
                            ? Colors.green
                            : Colors.blueGrey,
                        fontWeight: FontWeight.w600,
                      ),
                    ),

                    const SizedBox(height: 16),

                    const Text(
                      'After a successful DeepPCB inspection, Llama '
                      'automatically explains only the detected defect '
                      'classes. No manual prompt is needed.',
                      style: TextStyle(color: Colors.blueGrey),
                    ),

                    if (llamaResponse.isNotEmpty) ...[
                      const SizedBox(height: 16),
                      const Text(
                        'Llama AI Response',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(14),
                        decoration: BoxDecoration(
                          color: const Color(0xFFEAF4F5),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: SelectableText(llamaResponse),
                      ),
                    ],
                  ],
                ),
              ),

              const SizedBox(height: 24),

              if (selectedImage != null) ...[
                const Text(
                  'Selected PCB Image',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 12),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(18),
                  ),
                  child: Column(
                    children: [
                      ClipRRect(
                        borderRadius: BorderRadius.circular(12),
                        child: SizedBox(
                          width: double.infinity,
                          height: 260,
                          child: Stack(
                            fit: StackFit.expand,
                            children: [
                              Image.file(
                                selectedImage!,
                                fit: BoxFit.contain,
                                errorBuilder:
                                    (context, error, stackTrace) {
                                  return const Center(
                                    child: Text(
                                      'Unable to display image',
                                    ),
                                  );
                                },
                              ),
                              if (hasInspectionResult)
                                CustomPaint(
                                  painter: DefectOverlayPainter(
                                    detections: detections,
                                  ),
                                ),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(height: 12),
                      Row(
                        children: [
                          Expanded(
                            child: OutlinedButton.icon(
                              onPressed: isRunningInference
                                  ? null
                                  : showImageOptions,
                              icon: const Icon(Icons.refresh),
                              label: const Text('Replace'),
                            ),
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: OutlinedButton.icon(
                              onPressed:
                                  isRunningInference ? null : removeImage,
                              icon: const Icon(Icons.delete_outline),
                              label: const Text('Remove'),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 20),
              ],

              if (hasInspectionResult) ...[
                const Text(
                  'DeepPCB Inspection Results',
                  style: TextStyle(
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 12),
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(16),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment:
                            MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            'Total detected defects',
                            style: TextStyle(
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          Text(
                            '${detections.length}',
                            style: const TextStyle(
                              fontSize: 24,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF087E8B),
                            ),
                          ),
                        ],
                      ),
                      const Divider(height: 24),
                      if (detections.isEmpty)
                        const Text(
                          'No defects detected above the current '
                          'confidence threshold. This does not prove '
                          'the PCB is defect-free.',
                        )
                      else ...[
                        buildOverallSeverityCard(),
                        const SizedBox(height: 14),
                        ...List.generate(severityResults.length, (index) {
                          return buildInspectionReport(
                            severityResults[index],
                            detections[index],
                          );
                        }),
                      ],
                    ],
                  ),
                ),
                const SizedBox(height: 20),
              ],

              const Text(
                'DeepPCB Defect Classes',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 12),

              moduleCard(
                icon: Icons.electrical_services,
                title: 'Open',
                description: 'Open circuit defect detection.',
              ),
              moduleCard(
                icon: Icons.warning_amber,
                title: 'Short',
                description: 'Short circuit defect detection.',
              ),
              moduleCard(
                icon: Icons.blur_on,
                title: 'Mousebite',
                description: 'Mousebite defect detection.',
              ),
              moduleCard(
                icon: Icons.account_tree_outlined,
                title: 'Spur',
                description: 'Spur defect detection.',
              ),
              moduleCard(
                icon: Icons.device_hub,
                title: 'Spurious Copper',
                description: 'Spurious copper defect detection.',
              ),
              moduleCard(
                icon: Icons.circle_outlined,
                title: 'Pin Hole',
                description: 'Pin hole defect detection.',
              ),

              const SizedBox(height: 20),

              const Center(
                child: Text(
                  'OFFLINE DEEPPCB INSPECTION',
                  style: TextStyle(
                    color: Colors.blueGrey,
                    fontSize: 11,
                    letterSpacing: 1.5,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ------------------------------------------------------------
  // INSPECTION REPORT WIDGETS
  // ------------------------------------------------------------

  Widget buildOverallSeverityCard() {
    final overall =
        severityAssessmentService.overallSeverity(severityResults);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFEAF4F5),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          const Expanded(
            child: Text(
              'Overall PCB Severity',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
          ),
          Text(
            severityLabel(overall),
            style: TextStyle(
              fontWeight: FontWeight.bold,
              color: severityColor(overall),
            ),
          ),
        ],
      ),
    );
  }

  Widget buildInspectionReport(
    SeverityResult result,
    DeepPcbDetection detection,
  ) {
    final confidenceText =
        '${(result.confidence * 100).toStringAsFixed(1)}%';

    final category = confidenceCategory(result.confidence);

    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 14),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFFF8FAFC),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: const Color(0xFFE1E8EF),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'DeepPCB Inspection Result',
            style: TextStyle(
              fontSize: 17,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 14),
          reportLine('Defect', result.defectName),
          reportLine(
            'Location',
            'X:${result.x.round()}, Y:${result.y.round()}',
          ),
          reportLine(
            'Bounding Box',
            'X1:${detection.x1.round()}, '
            'Y1:${detection.y1.round()}, '
            'X2:${detection.x2.round()}, '
            'Y2:${detection.y2.round()}',
          ),
          reportLine(
            'Confidence',
            '$confidenceText ($category)',
            valueColor: confidenceColor(result.confidence),
          ),
          reportLine(
            'Severity',
            result.severityLabel,
            valueColor: severityColor(result.severity),
          ),
          reportLine('Status', result.status),
        ],
      ),
    );
  }
    Widget reportLine(
    String label,
    String value, {
    Color? valueColor,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 9),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 105,
            child: Text(
              label,
              style: const TextStyle(
                color: Colors.blueGrey,
                fontSize: 14,
              ),
            ),
          ),
          const Text(': '),
          Expanded(
            child: Text(
              value,
              style: TextStyle(
                fontWeight: FontWeight.w600,
                fontSize: 14,
                color: valueColor ?? const Color(0xFF263238),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget moduleCard({
    required IconData icon,
    required String title,
    required String description,
  }) {
    return Card(
      elevation: 0,
      color: Colors.white,
      margin: const EdgeInsets.only(bottom: 12),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
      ),
      child: ListTile(
        contentPadding: const EdgeInsets.all(14),
        leading: CircleAvatar(
          radius: 25,
          backgroundColor: const Color(0xFFE0F4F2),
          child: Icon(
            icon,
            color: const Color(0xFF087E8B),
          ),
        ),
        title: Text(
          title,
          style: const TextStyle(fontWeight: FontWeight.bold),
        ),
        subtitle: Padding(
          padding: const EdgeInsets.only(top: 5),
          child: Text(description),
        ),
        trailing: const Icon(
          Icons.arrow_forward_ios,
          size: 16,
        ),
      ),
    );
  }
}

class InspectionHistoryScreen extends StatelessWidget {
  final List<Map<String, dynamic>> records;

  const InspectionHistoryScreen({
    super.key,
    required this.records,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Inspection History'),
      ),
      body: records.isEmpty
          ? const Center(
              child: Text('No saved inspections yet.'),
            )
          : ListView.builder(
              padding: const EdgeInsets.all(12),
              itemCount: records.length,
              itemBuilder: (context, index) {
                final r = records[index];

                final date =
                    DateTime.tryParse('${r['inspectedAt']}')?.toLocal();

                final imagePath = '${r['imagePath'] ?? ''}';
                final imageFile = File(imagePath);

                return Card(
                  child: ListTile(
                    leading:
                        imagePath.isNotEmpty && imageFile.existsSync()
                            ? Image.file(
                                imageFile,
                                width: 58,
                                height: 58,
                                fit: BoxFit.cover,
                              )
                            : const Icon(Icons.image_not_supported),
                    title: Text(
                      '${r['totalDefects'] ?? 0} defect(s) • '
                      '${r['overallSeverity'] ?? 'Unknown'}',
                    ),
                    subtitle: Text(
                      '${date ?? 'Unknown date'}\n'
                      '${r['aiStatus'] ?? ''}',
                    ),
                    isThreeLine: true,
                    onTap: () {
                      showDialog(
                        context: context,
                        builder: (_) => AlertDialog(
                          title: const Text('Inspection Report'),
                          content: SingleChildScrollView(
                            child: SelectableText(
                              'Date: ${date ?? 'Unknown'}\n'
                              'Defects: ${r['totalDefects']}\n'
                              'Overall severity: ${r['overallSeverity']}\n'
                              'AI status: ${r['aiStatus']}\n\n'
                              '${r['aiReport'] ?? ''}',
                            ),
                          ),
                          actions: [
                            TextButton(
                              onPressed: () => Navigator.pop(context),
                              child: const Text('Close'),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                );
              },
            ),
    );
  }
}

// ------------------------------------------------------------
// CONFIDENCE AND SEVERITY HELPERS
// ------------------------------------------------------------

String confidenceCategory(double confidence) {
  final percentage = confidence * 100;

  if (percentage < 50) {
    return 'Low';
  } else if (percentage < 75) {
    return 'Medium';
  } else if (percentage < 90) {
    return 'High';
  } else {
    return 'Very High';
  }
}

Color confidenceColor(double confidence) {
  final percentage = confidence * 100;

  if (percentage < 50) {
    return Colors.red;
  } else if (percentage < 75) {
    return Colors.orange;
  } else if (percentage < 90) {
    return Colors.blue;
  } else {
    return Colors.green;
  }
}

String severityLabel(DefectSeverity severity) {
  switch (severity) {
    case DefectSeverity.low:
      return 'Low';
    case DefectSeverity.medium:
      return 'Medium';
    case DefectSeverity.high:
      return 'High';
    case DefectSeverity.critical:
      return 'Critical';
  }
}

Color severityColor(DefectSeverity severity) {
  switch (severity) {
    case DefectSeverity.low:
      return Colors.green;
    case DefectSeverity.medium:
      return Colors.orange;
    case DefectSeverity.high:
      return Colors.deepOrange;
    case DefectSeverity.critical:
      return Colors.red;
  }
}

// ------------------------------------------------------------
// DEEPPCB DEFECT COLORS
// ------------------------------------------------------------

Color defectColor(int classId) {
  const colors = [
    Colors.red,
    Colors.orange,
    Colors.purple,
    Colors.blue,
    Colors.green,
    Colors.pink,
  ];

  return colors[classId.clamp(0, colors.length - 1).toInt()];
}

// ------------------------------------------------------------
// DEEPPCB BOUNDING BOX PAINTER
// ------------------------------------------------------------

class DefectOverlayPainter extends CustomPainter {
  final List<DeepPcbDetection> detections;

  DefectOverlayPainter({
    required this.detections,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (detections.isEmpty) return;

    final imageWidth = detections.first.imageWidth.toDouble();
    final imageHeight = detections.first.imageHeight.toDouble();

    final scale = math.min(
      size.width / imageWidth,
      size.height / imageHeight,
    );

    final displayWidth = imageWidth * scale;
    final displayHeight = imageHeight * scale;

    final offsetX = (size.width - displayWidth) / 2;
    final offsetY = (size.height - displayHeight) / 2;

    for (final detection in detections) {
      final color = defectColor(detection.classId);

      final rect = Rect.fromLTRB(
        offsetX + detection.x1 * scale,
        offsetY + detection.y1 * scale,
        offsetX + detection.x2 * scale,
        offsetY + detection.y2 * scale,
      );

      final borderPaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5;

      canvas.drawRect(rect, borderPaint);

      final label =
          '${detection.className} '
          '${(detection.confidence * 100).toStringAsFixed(0)}%';

      final textPainter = TextPainter(
        text: TextSpan(
          text: label,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 11,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      );

      textPainter.layout(maxWidth: size.width);

      const horizontalPadding = 5.0;
      const verticalPadding = 3.0;

      final labelWidth =
          textPainter.width + horizontalPadding * 2;

      final labelHeight =
          textPainter.height + verticalPadding * 2;

      final labelLeft = rect.left.clamp(
        0.0,
        math.max(0.0, size.width - labelWidth),
      ).toDouble();

      final labelTop = (rect.top - labelHeight).clamp(
        0.0,
        math.max(0.0, size.height - labelHeight),
      ).toDouble();

      final labelRect = Rect.fromLTWH(
        labelLeft,
        labelTop,
        labelWidth,
        labelHeight,
      );

      canvas.drawRRect(
        RRect.fromRectAndRadius(
          labelRect,
          const Radius.circular(4),
        ),
        Paint()..color = color,
      );

      textPainter.paint(
        canvas,
        Offset(
          labelLeft + horizontalPadding,
          labelTop + verticalPadding,
        ),
      );
    }
  }

  @override
  bool shouldRepaint(
    covariant DefectOverlayPainter oldDelegate,
  ) {
    return oldDelegate.detections != detections;
  }
}
