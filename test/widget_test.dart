
import 'package:flutter_test/flutter_test.dart';
import 'package:industrial_defect_detection/main.dart';

void main() {
  testWidgets('DeepPCB application starts', (WidgetTester tester) async {
    await tester.pumpWidget(const DefectDetectionApp());

    expect(find.text('DeepPCB Inspector'), findsOneWidget);
    expect(find.text('DeepPCB Defect Detection'), findsOneWidget);
  });
}
