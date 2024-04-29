using UnityEngine;

public class CameraSwitcher : MonoBehaviour
{
    public Camera followCamera;
    public Camera onboardCamera;

    public Transform target; // Дрон

    void Start()
    {
        // Инициализация камер
        followCamera.enabled = true;
        onboardCamera.enabled = false;
    }

    void Update()
    {
        // Переключение между камерами по нажатию клавиши "C"
        if (Input.GetKeyDown(KeyCode.C))
        {
            SwitchCameras();
        }

        // Обновление положения камеры следования за дроном
        followCamera.transform.LookAt(target);
    }

    void SwitchCameras()
    {
        // Включение/выключение камер
        followCamera.enabled = !followCamera.enabled;
        onboardCamera.enabled = !onboardCamera.enabled;
    }
}