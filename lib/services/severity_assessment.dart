
enum DefectSeverity {
  low,
  medium,
  high,
  critical,
}

class SeverityResult {
  final String defectName;
  final double confidence;
  final double x;
  final double y;
  final DefectSeverity severity;
  final String status;

  const SeverityResult({
    required this.defectName,
    required this.confidence,
    required this.x,
    required this.y,
    required this.severity,
    required this.status,
  });

  String get severityLabel {
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
}

class SeverityAssessmentService {
  // DeepPCB-only severity rules.
  // These are project-defined provisional classifications.
  static const Map<String, DefectSeverity> severityRules = {
    'Open': DefectSeverity.high,
    'Short': DefectSeverity.critical,
    'Mousebite': DefectSeverity.medium,
    'Spur': DefectSeverity.medium,
    'Spurious Copper': DefectSeverity.high,
    'Pin Hole': DefectSeverity.medium,
  };

  SeverityResult assess({
    required String defectName,
    required double confidence,
    required double x,
    required double y,
  }) {
    final severity =
        severityRules[defectName] ?? DefectSeverity.medium;

    return SeverityResult(
      defectName: defectName,
      confidence: confidence,
      x: x,
      y: y,
      severity: severity,
      status: 'Defective',
    );
  }

  DefectSeverity overallSeverity(List<SeverityResult> results) {
    if (results.isEmpty) return DefectSeverity.low;

    const priority = {
      DefectSeverity.low: 1,
      DefectSeverity.medium: 2,
      DefectSeverity.high: 3,
      DefectSeverity.critical: 4,
    };

    return results
        .map((result) => result.severity)
        .reduce(
          (a, b) => priority[a]! >= priority[b]! ? a : b,
        );
  }
}
