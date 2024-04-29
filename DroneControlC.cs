using UnityEngine;
using System.Collections;
using System.IO;
using System.Diagnostics;
using System;
using System.Collections.Generic;
using System.Net.Sockets;
using UnityEngine.SceneManagement;
using System.Threading.Tasks;
using System.Linq; 
using UnityEngine.Events;

[Serializable]
public class CameraObjectPair
{
public Camera camera;
public GameObject gameObject;
public BoxCollider boxCollider; 
}

public class DroneControlC : MonoBehaviour
{
    public List<Rigidbody> drones = new List<Rigidbody>();
    public List<Camera> targetCameras = new List<Camera>();
    public List<CameraObjectPair> targetObjects = new List<CameraObjectPair>();
    public List<int> droneIDs = new List<int>();

    public int ForwardBackward = 10;
    public int UpDown = 10;
    public int Tilt = 5;
    public int FlyLeftRight = 50;
    
    public int time_for_restart = 90;

    private List<float> distanceToTerrainList = new List<float>();
    private List<bool> isInFieldOfViewList = new List<bool>();
    private List<float> rollList = new List<float>();
    private List<float> pitchList = new List<float>();
    private List<float> yawList = new List<float>();
    private List<List<float>> boundingBoxList = new List<List<float>>();


    private Vector3 DroneRotation;
    public float timeBetweenInputs = 0.2f;
    private float timer = 0f;
    float startTime;
    
    private TcpClient client;
    private NetworkStream stream;
    private StreamReader reader;
    private string pythonScriptPath = "/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/C#Script_tester.py";


    private Dictionary<int, int> action_number = new Dictionary<int, int>();

    bool[] actionVector = new bool[] { false, false, false, false };
    Dictionary<int, bool[]> action_dict = new Dictionary<int, bool[]>();

    bool restart = true;
    static bool initialized = false;
    bool TrueFalse = false;
    string filePath;
    
    float episode_mse;
    float episode_reward;
    
    float initialUpDown = 11f; 
    float UpDownf;

    private Dictionary<int, float> upDownTimers = new Dictionary<int, float>();
    private Dictionary<int, float[]> droneStates = new Dictionary<int, float[]>();
    private Dictionary<int, int> counters = new Dictionary<int, int>();
    private Dictionary<int, float[]> changes = new Dictionary<int, float[]>();
    private Dictionary<int, float[]> old_states = new Dictionary<int, float[]>();
    private Dictionary<int, float[]> new_states = new Dictionary<int, float[]>();
    private Dictionary<int, float> timer_drones = new Dictionary<int, float>();
    private Dictionary<int, float> velocity_x = new Dictionary<int, float>();
    private Dictionary<int, float> velocity_y = new Dictionary<int, float>();
    private Dictionary<int, float> velocity_z = new Dictionary<int, float>();
    private Dictionary<int, float> yaw = new Dictionary<int, float>();
    private Dictionary<int, float> roll = new Dictionary<int, float>();
    private Dictionary<int, float> pitch = new Dictionary<int, float>();
    private Dictionary<int, Vector3> initialRotations = new Dictionary<int, Vector3>();


    private List<float> withinTerr = new List<float>();

    List<float> withinBounds = new List<float>();

    private Vector3 initialPosition;
    private Dictionary<int, Vector3> initialPositions = new Dictionary<int, Vector3>();
    private Dictionary<int, Vector3> initialOffsets = new Dictionary<int, Vector3>();
    private Dictionary<int, Vector3> previousVelocity = new Dictionary<int, Vector3>();
    private Dictionary<int, Vector3> previousAngularVelocity = new Dictionary<int, Vector3>();


    void Start()
    {
        counters = new Dictionary<int, int>();
        action_number = new Dictionary<int, int>();
        changes = new Dictionary<int, float[]>();
        old_states = new Dictionary<int, float[]>();
        new_states = new Dictionary<int, float[]>();
        UpDownf = UpDown * 1f;

        foreach (int droneID in droneIDs)
        {
            List<float> rectList = new List<float>(4); // Изменение типа на List<float>
            for (int i = 0; i < 4; i++)
            {
                rectList.Add(-1f); // Замена Rect на float
            }
            boundingBoxList.Add(rectList);
        }

        // Задаем размеры словарей такими же, как у списка droneIDs
        foreach (int droneID in droneIDs)
        {
            previousVelocity[droneID] = Vector3.zero;
            previousAngularVelocity[droneID] = Vector3.zero;
            upDownTimers.Add(droneID, 0f);
            counters.Add(droneID, 0);
            action_number.Add(droneID, 0);
            changes.Add(droneID, new float[16]);
            old_states.Add(droneID, new float[16]);
            new_states.Add(droneID, new float[16]);
            withinBounds.Add(1.0f);
            droneStates.Add(droneID, new float[16]);
            initialPositions.Add(droneID, drones[droneID].transform.position);
            CameraObjectPair targetPair = targetObjects.FirstOrDefault(pair => pair.camera == targetCameras[droneID]);
            if (targetPair != null)
            {
                initialOffsets.Add(droneID, drones[droneID].transform.position - targetPair.gameObject.transform.position);
            }
            else
            {
                UnityEngine.Debug.LogWarning($"No corresponding gameObject found for drone {droneID}");
            }
            timer_drones.Add(droneID, 0f);
            velocity_x.Add(droneID, 0f);
            velocity_y.Add(droneID, 0f);
            yaw.Add(droneID, 0f);
            roll.Add(droneID, 0f);
            pitch.Add(droneID, 0f);    
            withinTerr.Add(1.0f);
            initialRotations.Add(droneID, drones[droneID].transform.localEulerAngles);  
        }

        action_dict.Add(0, new bool[] {false, false, false, false, false});
        action_dict.Add(1, new bool[] {false, true, true, false, false});
        action_dict.Add(2, new bool[] {true, false, false, true, false});
        action_dict.Add(3, new bool[] {false, false, true, true, false});
        action_dict.Add(4, new bool[] {true, true, false, false, false});

        action_dict.Add(5, new bool[] {true, true, true, true, false});
        action_dict.Add(6, new bool[] {true, true, true, true, true});



        startTime = Time.time;
        episode_mse = 0;
        episode_reward = 0;
    }

    
    float[] old_state = new float[16];
    float[] new_state = new float[16]; 
    //  Создаем очередь действий, которые будут выполнены в основном потоке
    private Queue<Action> mainThreadActions = new Queue<Action>();

    int[] actions = new int[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }; 

    float[] rewards = new float[] { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }; 
    float[] values = new float[] { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f  }; 

    string old_ = @"4.30285, 0, 0, 322.2589, 0, -4.142378, 0, 1, 93.74432, 119.0862, 120.1626, 126.251, 0.32
    9.735395, 0, 0, 322.2589, 0, -4.142378, 0, 1, 116.5685, 125.3742, 135.2241, 130.3223, 0.32
    22.56441, 0, 0, 322.2589, 0, -4.142378, 0, 1, 71.77246, 117.4743, 93.41431, 123.1841, 0.32
    40.44924, 0, 0, 322.2589, 0, -4.142378, 0, 1, 4.885284, 88.92899, 49.03054, 105.8172, 0.32
    27.24892, 0, 0, 322.2589, 0, -4.142378, 0, 1, 66.70301, 115.8441, 86.94145, 121.3843, 0.32
    27.24892, 0, 0, 322.2589, 0, -4.142378, 0, 1, 66.70301, 115.8441, 86.94145, 121.3843, 0.32
    4.30285, 0, 0, 322.2589, 0, -4.142378, 0, 1, 93.74432, 119.0862, 120.1626, 126.251, 0.32
    9.735395, 0, 0, 322.2589, 0, -4.142378, 0, 1, 116.5685, 125.3742, 135.2241, 130.3223, 0.32
    22.56441, 0, 0, 322.2589, 0, -4.142378, 0, 1, 71.77246, 117.4743, 93.41431, 123.1841, 0.32
    40.44924, 0, 0, 322.2589, 0, -4.142378, 0, 1, 4.885284, 88.92899, 49.03054, 105.8172, 0.32
    27.24892, 0, 0, 322.2589, 0, -4.142378, 0, 1, 66.70301, 115.8441, 86.94145, 121.3843, 0.32
    27.24892, 0, 0, 322.2589, 0, -4.142378, 0, 1, 66.70301, 115.8441, 86.94145, 121.3843, 0.32";

    string new_ = ""; // Переместите объявление переменной new_ сюда
    string act = null;
    
    float AngleDiff(float a1, float a2) 
    {
        float diff = Mathf.Repeat((a1 - a2), 360);
        if (diff > 180)
        {
            diff -= 360; 
        }
        return diff;
    }
    string DroneInfo;

    async void FixedUpdate()
    {
        timer += Time.deltaTime; 

        if (timer >= timeBetweenInputs)
        {
            timer = 0f; 
            try
            {

                List<float> distances = await CalculateDistanceToTerrainAsync(drones);
                List<float> withinTerr = await CheckDroneBoundsAsync(drones);
                boundingBoxList.Clear(); // Очистить список boundingBoxList перед добавлением новых значений

                new_ = "";
                foreach (var pair in drones.Select((drone, index) => (drone, index)))
                {
                    var Drone = pair.drone;
                    var droneIndex = pair.index;
                    CameraObjectPair targetPair = targetObjects[droneIndex];
                    List<float> droneBoundingBox = GetBoundingBox(targetPair.gameObject, targetPair.camera, targetPair.boxCollider);
                    boundingBoxList.Add(droneBoundingBox);
                    
                    Vector3 localVelocity = Drone.transform.InverseTransformDirection(Drone.velocity);
                    Vector3 localAngularVelocity = Drone.transform.InverseTransformDirection(Drone.angularVelocity);

                    Vector3 acceleration = (localVelocity - previousVelocity[droneIndex]) / timeBetweenInputs;
                    Vector3 angularAcceleration = (localAngularVelocity - previousAngularVelocity[droneIndex]) / timeBetweenInputs;

                    velocity_x[droneIndex] = localVelocity.x;
                    velocity_y[droneIndex] = localVelocity.y;
                    velocity_z[droneIndex] = localVelocity.z;

                    float angularVelocityX = localAngularVelocity.x;
                    float angularVelocityY = localAngularVelocity.y;
                    float angularVelocityZ = localAngularVelocity.z;

                    Vector3 currentRotation = Drone.transform.localEulerAngles;

                    float relYaw = AngleDiff(initialRotations[droneIndex].y, currentRotation.y);
                    float relPitch = AngleDiff(initialRotations[droneIndex].x, currentRotation.x);
                    float relRoll = AngleDiff(initialRotations[droneIndex].z, currentRotation.z);
                    
                    Vector3 dronePosition = Drone.transform.position;

                    previousVelocity[droneIndex] = localVelocity;
                    previousAngularVelocity[droneIndex] = localAngularVelocity;

                    if (droneIndex >= 0 && droneIndex < boundingBoxList.Count)
                    {
                        var droneBoundingBoxes = boundingBoxList[droneIndex];
                        string droneInfo = distances[droneIDs[droneIndex]].ToString("F4") + ", " +
                                        relRoll.ToString("F4") + ", " +
                                        relPitch.ToString("F4") + ", " +
                                        relYaw.ToString("F4") + ", " +
                                        velocity_x[droneIDs[droneIndex]].ToString("F4") + ", " +
                                        velocity_y[droneIDs[droneIndex]].ToString("F4") + ", " +
                                        velocity_z[droneIDs[droneIndex]].ToString("F4") + ", " +
                                        angularVelocityX.ToString("F4") + ", " + // Добавлена скорость вращения по оси X
                                        angularVelocityY.ToString("F4") + ", " + // Добавлена скорость вращения по оси Y
                                        angularVelocityZ.ToString("F4") + ", " + // Добавлена скорость вращения по оси Z
                                        string.Join(", ", droneBoundingBoxes.Select(x => x.ToString("F4"))) + ", " +
                                        timer_drones[droneIDs[droneIndex]].ToString("F4");
                        new_ += droneInfo + $", {acceleration.x.ToString("F4")}, {acceleration.y.ToString("F4")}, {acceleration.z.ToString("F4")}, {angularAcceleration.x.ToString("F4")}, {angularAcceleration.y.ToString("F4")}, {angularAcceleration.z.ToString("F4")}" +"\n";
                        DroneInfo  = droneInfo + "\n";
                    }
                    else
                    {
                        new_ += String.Join(", ", Enumerable.Repeat("0", 15));
                    }
                }
                string send_data = new_;
                Task<int[]> getActionTask = GetActionFromPythonAsync(send_data);
                actions  = await getActionTask;
            }
            catch (Exception ex)
            {
                UnityEngine.Debug.Log($"An error occurred while getting drone data: {ex.Message}");
                actions = new int[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }; 
            }
        }

            
        float[][] changes = new float[drones.Count][];
        Vector3[] droneRotations = drones.Select(drone => drone.transform.localEulerAngles).ToArray();
        foreach (var pair in drones.Select((drone, index) => (drone, index)))
        {
            var drone = pair.drone; 
            var index = pair.index;
            Vector3 DroneRotation = droneRotations[(int)index]; 
            int action = actions[(int)index]; 
            bool[] actionVector = action_dict[action];

            // if (actionVector[0] || actionVector[1] || actionVector[2] || actionVector[3])
            // {
            //     upDownTimers[(int)index] = timeBetweenInputs * 2; // Время уменьшения UpDown.
            //     UpDownf = initialUpDown;
            // }
            // else if (upDownTimers[(int)index] > 0)
            // {
            //     // Уменьшение UpDown с течением времени.
            //     upDownTimers[(int)index] -= Time.deltaTime;
            //     UpDownf = Mathf.Lerp(0, initialUpDown, upDownTimers[(int)index] / (timeBetweenInputs * 2));
            // }


            timer_drones[droneIDs[index]] += Time.deltaTime;
            
            // Получаем угловую скорость дрона
            Vector3 angularVelocity = drone.GetComponent<Rigidbody>().angularVelocity;

            // Демпфирование угловой скорости
            float damping = 0.6f; // Коэффициент демпфирования
            Vector3 dampingTorque = -angularVelocity * damping;
            drone.AddTorque(dampingTorque);

            if (DroneRotation.z > 10 && DroneRotation.z <= 180) { drone.AddRelativeTorque(0, 0, -10); }
            if (DroneRotation.z > 180 && DroneRotation.z <= 350) { drone.AddRelativeTorque(0, 0, 10); }
            if (DroneRotation.z > 1 && DroneRotation.z <= 10) { drone.AddRelativeTorque(0, 0, -3); }
            if (DroneRotation.z > 350 && DroneRotation.z < 359) { drone.AddRelativeTorque(0, 0, 3); }

            if (DroneRotation.x > 10 && DroneRotation.x <= 180) { drone.AddRelativeTorque(-10, 0, 0); }
            if (DroneRotation.x > 180 && DroneRotation.x <= 350) { drone.AddRelativeTorque(10, 0, 0); }
            if (DroneRotation.x > 1 && DroneRotation.x <= 10) { drone.AddRelativeTorque(-3, 0, 0); }
            if (DroneRotation.x > 350 && DroneRotation.x < 359) { drone.AddRelativeTorque(3, 0, 0); }

            // drone.AddForce(0, -9.8f, 0);
                    
            if (Input.GetKey(KeyCode.UpArrow)) { drone.AddRelativeForce(0, UpDownf, 0); }
            if (Input.GetKey(KeyCode.DownArrow)) { drone.AddRelativeForce(0, UpDownf / -1, 0); }

            if (actionVector[4])
            {
                // Применяем силу, точно равную и противоположную гравитации, чтобы дрон "завис"
                drone.AddForce(new Vector3(0, 11f * drone.mass, 0));

                // Плавное замедление скорости
                Vector3 targetVelocity = Vector3.zero;  // Целевая скорость - ноль
                Vector3 targetAngularVelocity = Vector3.zero;  // Целевая угловая скорость - ноль

                // Вычисляем новую скорость и угловую скорость, приближая текущие значения к нулю
                drone.velocity = Vector3.Lerp(drone.velocity, targetVelocity, Time.deltaTime / timeBetweenInputs * 1f);
                drone.angularVelocity = Vector3.Lerp(drone.angularVelocity, targetAngularVelocity, Time.deltaTime / timeBetweenInputs*1f);
            }
            else
            {
                // Применение сил и вращения с учетом нового значения UpDown.
                if (actionVector[0]) {
                    drone.AddRelativeForce(ForwardBackward, UpDownf, -ForwardBackward);
                    drone.AddRelativeTorque(-0.5f * Tilt, Tilt, -0.5f * Tilt); // Усиленный поворот налево
                }
                if (actionVector[1]) {
                    drone.AddRelativeForce(-ForwardBackward, UpDownf, -ForwardBackward);
                    drone.AddRelativeTorque(-0.5f * Tilt, -Tilt, 0.5f * Tilt); // Усиленный поворот направо
                }
                if (actionVector[2]) {
                    drone.AddRelativeForce(ForwardBackward, UpDownf, ForwardBackward);
                    drone.AddRelativeTorque(0.5f * Tilt, -Tilt, -0.5f * Tilt);
                }
                if (actionVector[3]) {
                    drone.AddRelativeForce(-ForwardBackward, UpDownf, ForwardBackward);
                    drone.AddRelativeTorque(0.5f * Tilt, Tilt, 0.5f * Tilt);
                }
            }

            Terrain terrain = Terrain.activeTerrain;
            Vector3 terrainSize = terrain.terrainData.size;
            Vector3 dronePosition = drone.position;
            bool isWithinBounds = dronePosition.x >= 0 && dronePosition.x <= terrainSize.x &&
                                dronePosition.y >= 0 && dronePosition.y <= terrainSize.y &&
                                dronePosition.z >= 0 && dronePosition.z <= terrainSize.z;

            if (!isWithinBounds || (DroneRotation.z > 175 && DroneRotation.z < 195) || (timer_drones[droneIDs[index]] > time_for_restart)) // || (index >= 0 && index < boundingBoxList.Count && boundingBoxList[(int)index][0] == -1))
            {
                CameraObjectPair targetPair = targetObjects.FirstOrDefault(pair => pair.camera == targetCameras[droneIDs[(int)index]]);
                if (targetPair != null && initialOffsets.TryGetValue(droneIDs[(int)index], out Vector3 initialOffset))
                {
                    drone.transform.position = targetPair.gameObject.transform.position + initialOffset;
                    drone.velocity = Vector3.zero;
                    drone.angularVelocity = Vector3.zero;
                    // Установка исходного угла наклона дрона
                    if (initialRotations.TryGetValue(droneIDs[(int)index], out Vector3 initialRotation))
                    {
                        drone.transform.localRotation = Quaternion.Euler(initialRotation);
                    }
                    else
                    {
                        UnityEngine.Debug.LogWarning($"Failed to reset rotation for drone {droneIDs[(int)index]}");
                    }
                }
                else
                {
                    UnityEngine.Debug.LogWarning($"Failed to reset position for drone {droneIDs[(int)index]}");
                }

                timer_drones[droneIDs[(int)index]] = 0f;
            }
        }
    }



    async Task<int[]> GetActionFromPythonAsync(string send_data)
    {
        string tempFile = Path.Combine(Path.GetTempPath(), $"tempfile.txt");

        await File.WriteAllTextAsync(tempFile, send_data);
        ProcessStartInfo start = new ProcessStartInfo();
        start.FileName = "/Users/sargisvardanyan/anaconda3/envs/graduate_env/bin/python";
        start.Arguments = $"/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/C#Script_tester.py \"{tempFile}\"";
        start.UseShellExecute = false;
        start.RedirectStandardOutput = true;
        start.RedirectStandardError = true;

        using (Process process = Process.Start(start))
        {
            string result = await process.StandardOutput.ReadToEndAsync();
            string error = await process.StandardError.ReadToEndAsync();
            UnityEngine.Debug.Log("Result: " + result );

            if (!string.IsNullOrEmpty(error))
            {   
                UnityEngine.Debug.LogError("Ошибка при вызове пайтон кода: " + error);
                return (null);
            }

            string[] parts = result.Split(new[] { '[', ']' }, StringSplitOptions.RemoveEmptyEntries);

            string actionPart = parts[0];
            string[] actionStrings = actionPart.Split(',');
            int[] actions = Array.ConvertAll(actionStrings, int.Parse);

            return (actions);
        }
    }

       
    void WriteDataToFile(float[] droneInfoArray, Rect boundingBox, int actionNumber, float startTime, int droneID)
    {
        filePath = $"/Users/sargisvardanyan/PycharmProjects/Quadcopter_RL_car_detection/results{droneID}.csv";
        float currentTime = Time.time - startTime;
        string data = $"{droneInfoArray[0]}, {droneInfoArray[1]}, {droneInfoArray[2]}, {droneInfoArray[3]}, {droneInfoArray[4]}, {droneInfoArray[5]}, {droneInfoArray[6]}, {droneInfoArray[7]}, {boundingBox.x}, {boundingBox.y}, {boundingBox.width}, {boundingBox.height}, {actionNumber}, {currentTime}\n";
        // UnityEngine.Debug.Log("Bounding Box: " + data);

        if (!File.Exists(filePath))
        {
            string header = "DisTerr,Roll,Pitch,Yaw,RotX,RotY,RotZ,Out_of_Terrain_TF,Bounding_box_X,Bounding_box_Y,Bounding_box_W,Bounding_box_H, action, time\n";
            File.AppendAllText(filePath, header);
        }

        File.AppendAllText(filePath, data);
    }

    void RestartLevel(int droneID)
    {
        if (initialPositions.ContainsKey(droneID))
        {
            Rigidbody drone = drones[droneID];
            drone.transform.position = initialPositions[droneID];
            drone.transform.rotation = Quaternion.identity;
            drone.velocity = Vector3.zero;
            initialPositions[droneID] = drone.transform.position; // Обновляем начальную позицию
            // startTime = Time.time;
        }
        else
        {
            UnityEngine.Debug.LogError("Начальная позиция для дрона с индексом " + droneID + " не определена.");
        }
    }


    // Изменить тип возвращаемого значения на List<float>
    List<float> GetBoundingBox(GameObject obj, Camera camera, BoxCollider boxCollider)
    {
        Bounds bounds = boxCollider.bounds;
        Vector3[] corners = new Vector3[8];
        corners[0] = new Vector3(bounds.min.x, bounds.min.y, bounds.min.z);
        corners[1] = new Vector3(bounds.min.x, bounds.min.y, bounds.max.z);
        corners[2] = new Vector3(bounds.min.x, bounds.max.y, bounds.min.z);
        corners[3] = new Vector3(bounds.min.x, bounds.max.y, bounds.max.z);
        corners[4] = new Vector3(bounds.max.x, bounds.min.y, bounds.min.z);
        corners[5] = new Vector3(bounds.max.x, bounds.min.y, bounds.max.z);
        corners[6] = new Vector3(bounds.max.x, bounds.max.y, bounds.min.z);
        corners[7] = new Vector3(bounds.max.x, bounds.max.y, bounds.max.z);

        List<Vector3> screenCorners = new List<Vector3>();
        // Использовать метод GeometryUtility.TestPlanesAABB для проверки видимости
        bool isVisible = GeometryUtility.TestPlanesAABB(GeometryUtility.CalculateFrustumPlanes(camera), bounds);

        foreach (var corner in corners)
        {
            Vector3 screenCorner = camera.WorldToScreenPoint(corner);
            screenCorners.Add(screenCorner);
        }
        if (isVisible)
        {
            float minX = float.MaxValue;
            float maxX = float.MinValue;
            float minY = float.MaxValue;
            float maxY = float.MinValue;

            foreach (var screenCorner in screenCorners)
            {
                minX = Mathf.Min(minX, screenCorner.x);
                maxX = Mathf.Max(maxX, screenCorner.x);
                minY = Mathf.Min(minY, screenCorner.y);
                maxY = Mathf.Max(maxY, screenCorner.y);
            }
        minX = Mathf.Clamp(minX, 0f, 256f);
        minY = Mathf.Clamp(minY, 0f, 256f);
        maxX = Mathf.Clamp(maxX, 0f, 256f);
        maxY = Mathf.Clamp(maxY, 0f, 256f);

        float width = maxX - minX;
        float height = maxY - minY;

        width = Mathf.Clamp(width, 0f, 256f);
        height = Mathf.Clamp(height, 0f, 256f);

            return new List<float>() { minX, minY, width, height };
        }
        else
        {
            // Возвращать список из четырех отрицательных значений
            return new List<float>() { -1f, -1f, -1f, -1f };
        }
    }



    bool IsInFieldOfView(Vector3 point, Camera camera)
    {
        Vector3 viewportPoint = camera.ScreenToViewportPoint(point);
        return viewportPoint.x >= 0 && viewportPoint.x <= 1 && viewportPoint.y >= 0 && viewportPoint.y <= 1;
    }


    async Task RunInMainThread(Action action)
    {
        var tcs = new TaskCompletionSource<bool>();
        UnityAction wrapperAction = () =>
        {
            action();
            tcs.SetResult(true);
        };
        await tcs.Task;
    }

    async Task<List<float>> CheckDroneBoundsAsync(List<Rigidbody> drones)
    {
        List<float> results = new List<float>();
        await Task.Run(() =>
        {
            foreach (var drone in drones)
            {
                mainThreadActions.Enqueue(() =>
                {
                    Terrain terrain = Terrain.activeTerrain;
                    Vector3 terrainSize = terrain.terrainData.size;
                    Vector3 dronePosition = drone.position;

                    float isWithinBounds = dronePosition.x >= 0 && dronePosition.x <= terrainSize.x &&
                                            dronePosition.y >= 0 && dronePosition.y <= terrainSize.y &&
                                            dronePosition.z >= 0 && dronePosition.z <= terrainSize.z ? 1.0f : 0.0f;
                    results.Add(isWithinBounds);

                });
            }
        });

        return results;
    }

    async Task<List<float>> CalculateDistanceToTerrainAsync(List<Rigidbody> drones)
    {
        List<float> results = new List<float>();
        await Task.Run(() =>
        {
            foreach (var drone in drones)
            {
                // Добавляем действие в очередь, чтобы вычислить расстояние до местности в основном потоке
                mainThreadActions.Enqueue(() =>
                {
                    Terrain terrain = Terrain.activeTerrain;
                    if (terrain != null)
                    {
                        Vector3 dronePosition = drone.position;
                        float terrainHeight = terrain.SampleHeight(dronePosition);

                        // Рассчитываем вертикальное расстояние между дроном и поверхностью местности
                        float distanceToTerrain = dronePosition.y - terrainHeight;
                        results.Add(distanceToTerrain);
                    }
                    else
                    {
                        results.Add(0f); // Если местность не найдена, предполагаем, что расстояние равно 0
                    }
                });
            }
        });
        return results;
    }
    
    async Task<List<float>> GetDroneRollAsync(List<Rigidbody> drones)
    {
        List<float> results = new List<float>();
        await Task.Run(() =>
        {
            foreach (var drone in drones)
            {
                // Добавляем действие в очередь, чтобы получить угол крена в основном потоке
                mainThreadActions.Enqueue(() =>
                {
                    float roll = drone.transform.rotation.eulerAngles.z;
                    results.Add(roll);
                });
            }
        });
        return results;
    }

    async Task<List<float>> GetDronePitchAsync(List<Rigidbody> drones)
    {
        List<float> results = new List<float>();
        await Task.Run(() =>
        {
            foreach (var drone in drones)
            {
                mainThreadActions.Enqueue(() =>
                {
                    float pitch = drone.transform.rotation.eulerAngles.x;
                    results.Add(pitch);
                });
            }
        });
        return results;
    }

    async Task<List<float>> GetDroneYawAsync(List<Rigidbody> drones)
    {
        List<float> results = new List<float>();
        await Task.Run(() =>
        {
            foreach (var drone in drones)
            {
                // Добавляем действие в очередь, чтобы получить угол рыскания в основном потоке
                mainThreadActions.Enqueue(() =>
                {
                    float yaw = drone.transform.rotation.eulerAngles.y;
                    results.Add(yaw);
                });
            }
        });
        return results;
    }

    private void Update()
    {
        while (mainThreadActions.Count > 0)
        {
            // Получаем первый элемент из очереди
            var action = mainThreadActions.Dequeue();
            if (action != null)
            {
                action.Invoke();
            }
        }
    }

}

