class Dish {
  final String koreanName;
  final String englishName;
  final String description;
  final String why;
  final List<String> imageUrls;

  Dish({
    required this.koreanName,
    required this.englishName,
    required this.description,
    required this.why,
    required this.imageUrls,
  });

  factory Dish.fromJson(Map<String, dynamic> json, List<String> imageUrls) {
    return Dish(
      koreanName: json['korean_name'],
      englishName: json['english_name'],
      description: json['description'],
      why: json['why'],
      imageUrls: imageUrls,
    );
  }
}
