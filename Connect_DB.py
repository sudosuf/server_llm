import sqlite3 as sql
import psycopg2


def connecting(text):
    connection = psycopg2.connect(
        database="Test_24_04_24",
        user="postgres",
        password="toor",
        host="localhost",
        port="5432"
    )
    cursor = connection.cursor()

    # Выполняем SELECT запрос
    cursor.execute(text)

    # Получаем результаты SELECT запроса
    results = cursor.fetchall()

    # Выводим результаты
    for row in results:
        print(row)

        # Закрываем курсор и соединение
    cursor.close()
    connection.close()
    return results
