# NoIT: Word Cloud Service (WCS) Project

![Word Cloud Service Logo](path/to/logo.png)

## Содержание

- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Описание

Представляем проект **Word Cloud Service (WCS)** от команды NoIT. Проект создан в рамках хакатона Nuclear Hack Autumn 2024 (НИЯУ МИФИ). Данный сервис позволяет получать интеллекутальное облако слов на основе ответов пользователей. ИИ, на котором основан сервис, автоматически выбирет наиболее важные слова и словосочетания, очищает их от нецензурной и вредной лексики и возвращает особенное облако слов виде персональной маски.

## Функции

- **Динамическая генераци облака слов**: Возможность создания облаков слов и любых наборов строк.
- **Кастомный внешний вид**: Есть возможность изменения маски для облака слов.
- **Скачиваемое облако слов**: Можно сохранить облако слов в формате изображения.
- **Поддержка нескольких языков**: Сервис обрабатывает как английский, так и русский язык с одинаковой эффективностью.

## Стек технологий

- **Frontend**: 
  - JS (Flask)

- **Backend**: 
  - Python (Flask)

- **ML**:
  - RuBERT
  - Pymorphy
  - SpaCy

## Состав команды

- Артемий Терещенко (embeddings + clusterizing)
- Карпова Юлия (data preprocessing + frontend)
- Гавриленко Андрей (backend)
- Сидоров Алексей (data preproceesing) 

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
   ```

3. **Run the application**:
   - Start the backend server:
     ```bash
     cd server
     npm start
     ```
   - Start the frontend application:
     ```bash
     cd client
     npm start
     ```

5. **Access the application**:
   Open your browser and navigate to `http://localhost:3000`.

## Использование

1. **Creating a Word Cloud**:
   - Navigate to the "Create" page.
   - Input your text in the provided text area.
   - Customize the appearance using the options available.
   - Click on "Generate Word Cloud" to see the result.

2. **Saving and Downloading**:
   - After generating a word cloud, you can save it to your account or download it directly.

3. **API Usage**:
   - To generate a word cloud via the API, send a POST request to `/api/wordcloud` with the text data in the body.

   Example:
   ```bash
   curl -X POST http://localhost:5000/api/wordcloud -H "Content-Type: application/json" -d '{"text": "your text here"}'
   ```

## Лицензия

Проект разрабатан под лицензией MIT. Смотреть файл LICENSE.md для подробностей.

## Благодарности

- Команда благодарит организаторов за проведение данного мероприятия.