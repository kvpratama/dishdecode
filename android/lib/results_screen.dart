import 'package:flutter/material.dart';
import 'dish_model.dart';

class ResultsScreen extends StatelessWidget {
  final List<Dish> dishes;

  const ResultsScreen({super.key, required this.dishes});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Results')),
      body: ListView.builder(
        itemCount: dishes.length,
        itemBuilder: (context, index) {
          final dish = dishes[index];
          return Card(
            margin: const EdgeInsets.all(8.0),
            child: Padding(
              padding: const EdgeInsets.all(8.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    '${dish.koreanName} / ${dish.englishName}',
                    style: const TextStyle(
                      fontSize: 18.0,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 8.0),
                  Text(dish.description),
                  const SizedBox(height: 8.0),
                  Text(
                    'Why you should try it: ${dish.why}',
                    style: const TextStyle(fontStyle: FontStyle.italic),
                  ),
                  const SizedBox(height: 8.0),
                  SizedBox(
                    height: 150,
                    child: ListView.builder(
                      scrollDirection: Axis.horizontal,
                      itemCount: dish.imageUrls.length,
                      itemBuilder: (context, index) {
                        return Padding(
                          padding: const EdgeInsets.only(right: 8.0),
                          child: Image.network(dish.imageUrls[index]),
                        );
                      },
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}
