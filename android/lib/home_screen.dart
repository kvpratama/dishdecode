import 'dart:io';

import 'api_service.dart';
import 'dish_model.dart';
import 'results_screen.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _isLoading = false;
  final ApiService _apiService = ApiService();

  Future<void> _takePicture() async {
    final picker = ImagePicker();
    final image = await picker.pickImage(source: ImageSource.camera);
    if (image != null) {
      _processImage(File(image.path));
    }
  }

  Future<void> _uploadImage() async {
    final picker = ImagePicker();
    final image = await picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      _processImage(File(image.path));
    }
  }

  Future<void> _processImage(File imageFile) async {
    setState(() {
      _isLoading = true;
    });

    try {
      final bytes = await imageFile.readAsBytes();
      final result = await _apiService.uploadImage(bytes);
      final dishes =
          (result['recommended_dishes'] as List)
              .map(
                (dishJson) => Dish.fromJson(
                  dishJson,
                  result['dish_images'][dishJson['korean_name']].cast<String>(),
                ),
              )
              .toList();

      if (mounted) {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => ResultsScreen(dishes: dishes),
          ),
        );
      }
    } catch (e) {
      // TODO: Handle error
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('DishDecode')),
      body:
          _isLoading
              ? const Center(child: CircularProgressIndicator())
              : Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: <Widget>[
                    ElevatedButton(
                      onPressed: _takePicture,
                      child: const Text('Take a Picture'),
                    ),
                    const SizedBox(height: 20),
                    ElevatedButton(
                      onPressed: _uploadImage,
                      child: const Text('Upload an Image'),
                    ),
                  ],
                ),
              ),
    );
  }
}
