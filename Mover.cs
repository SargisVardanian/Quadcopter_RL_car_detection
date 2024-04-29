using UnityEngine;

public class SimpleMover : MonoBehaviour {
    public float moveSpeed = 5f;
    // Задаём диапазон, в пределах которого будет происходить случайное изменение наклона
    public float tiltRange = 5f;

    void Update() {
        // Постоянное движение вперёд
        transform.Translate(Vector3.forward * moveSpeed * Time.deltaTime);

        // Применение случайного наклона
        if (Random.Range(0, 10) > 8) { // Условие для того, чтобы наклон не применялся каждый кадр
            float tilt = Random.Range(-tiltRange, tiltRange); // Выбор случайного значения наклона
            transform.Rotate(0, tilt * Time.deltaTime, 0); // Применение наклона по оси Y
        }
    }
}