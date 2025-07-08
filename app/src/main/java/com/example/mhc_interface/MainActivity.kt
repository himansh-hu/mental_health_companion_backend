import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

void main() {
    runApp(MentalHealthApp());
}

class MentalHealthApp extends StatelessWidget {
    @override
    Widget build(BuildContext context) {
        return MaterialApp(
            debugShowCheckedModeBanner: false,
        theme: ThemeData(
        primarySwatch: Colors.blue,
        textTheme: GoogleFonts.latoTextTheme(),
        ),
        home: HomeScreen(),
        );
    }
}

class HomeScreen extends StatelessWidget {
    @override
    Widget build(BuildContext context) {
        return Scaffold(
            body: Stack(
                    children: [
            Container(
                decoration: BoxDecoration(
                        gradient: LinearGradient(
                        colors: [Colors.blue.shade200, Colors.purple.shade300],
            begin: Alignment.topLeft,
        end: Alignment.bottomRight,
        ),
        ),
        ),
        Column(
            children: [
            SizedBox(height: 80),
        Text(
            'Mental Health Companion',
            style: TextStyle(
                    fontSize: 24,
        fontWeight: FontWeight.bold,
        color: Colors.white,
        ),
        ),
        SizedBox(height: 20),
        Expanded(
            child: GridView.count(
                padding: EdgeInsets.all(20),
        crossAxisCount: 2,
        mainAxisSpacing: 20,
        crossAxisSpacing: 20,
        children: [
        featureCard(Icons.chat, 'Chatbot'),
        featureCard(Icons.mood, 'Mood Tracker'),
        featureCard(Icons.self_improvement, 'Self-Help'),
        featureCard(Icons.help_outline, 'FAQ'),
        featureCard(Icons.phone, 'Emergency Help'),
        featureCard(Icons.settings, 'Settings'),
        ],
        ),
        ),
        ],
        ),
        ],
        ),
        );
    }

    Widget featureCard(IconData icon, String title) {
        return Card(
            elevation: 5,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        color: Colors.white.withOpacity(0.9),
        child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
        Icon(icon, size: 50, color: Colors.blueAccent),
        SizedBox(height: 10),
        Text(
            title,
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.w500),
        ),
        ],
        ),
        );
    }
}
