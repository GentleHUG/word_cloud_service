# NoIT: Сервис Облаков Слов (WCS)

# TODO: Добавить логотип
# TODO: Добавить requirements.txt

![Логотип Сервиса Облаков Слов](path/to/logo.png)

## Содержание

- [Введение](#введение)
- [Особенности](#особенности)
- [Используемые технологии](#используемые-технологии)
- [Состав команды](#состав-команды)
- [Установка](#установка)
- [Использование](#использование)
- [Лицензия](#лицензия)
- [Благодарности](#благодарности)

## Введение

Добро пожаловать в проект **Сервис Облаков Слов (WCS)** от команды NoIT! Этот проект был разработан в рамках хакатона Nuclear Hack Autumn 2024 (НИЯУ МИФИ). С помощью нашего сервиса вы можете создавать уникальные облака слов на основе ответов пользователей. Искусственный интеллект, лежащий в основе сервиса, автоматически выделяет наиболее значимые слова и фразы, очищает их от ненормативной лексики и формирует персонализированное облако слов в виде маски.

## Особенности

- **Динамическая генерация облаков слов**: Создавайте облака слов из любых наборов строк в реальном времени.
- **Кастомный внешний вид**: Настраивайте маску для облака слов по своему вкусу.
- **Скачиваемое облако слов**: Сохраняйте облака слов в формате изображения для дальнейшего использования.
- **Поддержка нескольких языков**: Эффективная обработка как русского, так и английского языков.

## Используемые технологии

- **Frontend**: 
  - JavaScript (Flask)

- **Backend**: 
  - Python (Flask)

- **Машинное обучение**:
  - RuBERT
  - Pymorphy
  - SpaCy
  - HDBSCAN

## Состав команды

- Артемий Терещенко (векторизация + кластеризация)
- Карпова Юлия (предобработка данных + фронтенд)
- Гавриленко Андрей (бэкенд)
- Сидоров Алексей (предобработка данных)

## Установка

Чтобы запустить сервис локально, выполните следующие шаги:

1. **Склонируйте репозиторий**:
   ```bash
   git clone https://github.com/GentleHUG/word_cloud_service.git
   cd word_cloud_service
   ```

2. **Установите зависимости**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Запустите приложение**:
   ```bash
   python app.py
   ```

4. **Доступ к сервису**:
   Откройте браузер и перейдите на `http://localhost:8000`.

## Использование

...

## Благодарности

- Мы выражаем благодарность организаторам за проведение этого замечательного мероприятия.

---

# NoIT: Word Cloud Service (WCS) Project

![Word Cloud Service Logo](path/to/logo.png)

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Team members](#team-members)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

Welcome to the **Word Cloud Service (WCS)** project by the NoIT team! This project was developed during the Nuclear Hack Autumn 2024 hackathon (NRNU MEPhI). With our service, you can create unique word clouds based on user responses. The artificial intelligence behind the service automatically selects the most significant words and phrases, cleans them of inappropriate language, and generates a personalized word cloud in the form of a mask.

## Features

- **Dynamic Word Cloud Generation**: Create word clouds from any set of strings in real-time.
- **Custom Appearance**: Customize the mask for the word cloud to your liking.
- **Downloadable Word Cloud**: Save word clouds as image files for further use.
- **Multilingual Support**: Efficient processing of both Russian and English languages.

## Technologies Used

- **Frontend**: 
  - JavaScript (Flask)

- **Backend**: 
  - Python (Flask)

- **Machine Learning**:
  - RuBERT
  - Pymorphy
  - SpaCy
  - HDBSCAN

## Team Members

- Artemiy Tereshchenko (embeddings + clustering)
- Yulia Karpova (data preprocessing + frontend)
- Andrey Gavrilenko (backend)
- Alexey Sidorov (data preprocessing)

## Installation

To run the service locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/GentleHUG/word_cloud_service.git
   cd word_cloud_service
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the service**:
   Open your browser and go to `http://localhost:8000`.

## Usage

...

## Acknowledgments

- We would like to thank the organizers for hosting this wonderful event and providing us with the opportunity to showcase our work.
