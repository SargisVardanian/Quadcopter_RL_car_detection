using UnityEngine;
using System.Collections;

public class ResetPosition : MonoBehaviour
{
    public Vector3 initialPosition;
    private float timer = 0.0f;
    public float resetTime = 16.0f;

    void Start()
    {
        // Сохраняем изначальное положение объекта
        initialPosition = transform.position;
    }

    void Update()
    {
        // Обновляем таймер каждый кадр
        timer += Time.deltaTime;

        // Проверяем, прошло ли 16 секунд
        if (timer >= resetTime)
        {
            // Возвращаем объект в изначальное положение
            transform.position = initialPosition;
            // Сбрасываем таймер
            timer = 0.0f;
        }
    }
}
