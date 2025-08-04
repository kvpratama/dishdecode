import 'dart:io';
import 'dart:async'; // For TimeoutException
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:typed_data'; // Import for Uint8List

class ApiService {
  Future<Map<String, dynamic>> uploadImage(Uint8List imageBytes) async {
    try {
      // Simulate a network request
      await Future.delayed(const Duration(seconds: 2));
      final headers = <String, String>{'Content-Type': 'application/json'};

      final body = r'{}';
      String? threadId;
      String? assistantId;

      print('Sending request to: http://10.0.2.2:2024/threads');
      // print('Request body: $body');

      final response_thread = await http.post(
        Uri.parse('http://10.0.2.2:2024/threads'),
        headers: headers,
        body: body,
      );

      // print('Response status: ${response_thread.statusCode}');
      // print('Response body: ${response_thread.body}');

      // If we get here, the request was successful
      if (response_thread.statusCode >= 200 &&
          response_thread.statusCode < 300) {
        try {
          // Decode the JSON string into a Map
          final Map<String, dynamic> responseData = jsonDecode(
            response_thread.body,
          );

          // Now you can safely access the 'thread_id'
          threadId = responseData['thread_id'];

          print('Thread ID: $threadId');
        } catch (e) {
          print('Error decoding JSON: $e');
        }
      } else {
        print('Request failed with status: ${response_thread.statusCode}');
      }

      final headers_assistant = <String, String>{
        'Content-Type': 'application/json',
      };

      final body_assistant =
          r'{"assistant_id":"","graph_id":"main_graph","config":{},"metadata":{},"if_exists":"raise","name":"","description":null}';

      final response_assistant = await http.post(
        Uri.parse('http://10.0.2.2:2024/assistants'),
        headers: headers_assistant,
        body: body_assistant,
      );
      // print(response_assistant.body);

      if (response_assistant.statusCode >= 200 &&
          response_assistant.statusCode < 300) {
        try {
          // Decode the JSON string into a Map
          final Map<String, dynamic> responseData = jsonDecode(
            response_assistant.body,
          );

          // Now you can safely access the 'thread_id'
          assistantId = responseData['assistant_id'];

          print('Assistant ID: $assistantId');
        } catch (e) {
          print('Error decoding JSON: $e');
        }
      } else {
        print('Request failed with status: ${response_assistant.statusCode}');
      }

      final headers_run = <String, String>{'Content-Type': 'application/json'};
      final String base64Image = base64Encode(imageBytes);
      final input_data = {"image": base64Image, "max_size": 640};
      final body_run_map = {"assistant_id": assistantId, "input": input_data};
      final response_run = await http.post(
        Uri.parse('http://10.0.2.2:2024/threads/$threadId/runs/wait'),
        headers: headers_run,
        body: jsonEncode(body_run_map),
      );
      print(response_run.body);
      return jsonDecode(response_run.body);
    } on TimeoutException catch (e) {
      print('Timeout error: $e');
      rethrow; // or return mock data
    } on SocketException catch (e) {
      print('Network error: $e');
      print('Make sure the server is running and accessible at 10.0.2.2:2024');
      rethrow; // or return mock data
    } catch (e) {
      print('Unexpected error: $e');
      rethrow; // or return mock data
    }
  }
}
