using System;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class TrajectoryFollower : MonoBehaviour
{
    [Header("Trajectory Source")]
    [Tooltip("If true, it will load from the JSON file. If false, it will try to get the path from a TrajectoryGenerator component on this object.")]
    public bool loadFromJson = true;
    public string jsonFileName = "optimal_trajectory.json";

    [Header("Movement Settings")]
    public float speed = 5f;

    [Header("Rotation Settings")]
    [Tooltip("If true, the object will use the exact rotation stored in the waypoint. If false, it will actively steer towards the next waypoint.")]
    public bool useStoredRotation = true;

    [Tooltip("Degrees per second the car can rotate (Used only if useStoredRotation is false).")]
    public float turnSpeedDegrees = 90f;

    [Tooltip("Distance to a waypoint before advancing to the next one.")]
    public float waypointThreshold = 0.3f;

    private List<TrajectoryGenerator.TrajectoryPoint> waypoints = new List<TrajectoryGenerator.TrajectoryPoint>();
    private int currentWaypoint = 0;

    void Start()
    {
        if (loadFromJson)
        {
            Vector3 startingPosition = transform.position;
            LoadTrajectoryFromJson(startingPosition);
        }
        else
        {
            TrajectoryGenerator generator = GetComponent<TrajectoryGenerator>();
            if (generator != null)
            {
                waypoints = generator.GetWaypoints();
                Debug.Log($"{gameObject.name}: Loaded {waypoints.Count} waypoints from Generator.");
            }
            else
            {
                Debug.LogError($"{gameObject.name} is set to use Generator, but no TrajectoryGenerator component was found!");
            }
        }

        if (waypoints.Count > 0)
        {
            currentWaypoint = 1; 
        }
    }

    void LoadTrajectoryFromJson(Vector3 startOffsetPos)
    {
        string filePath = Path.Combine(Application.dataPath, jsonFileName);

        if (!File.Exists(filePath))
        {
            Debug.LogError($"Could not find trajectory file:\n{filePath}");
            return;
        }

        string json = File.ReadAllText(filePath);
        RootTrajectoryData root = JsonUtility.FromJson<RootTrajectoryData>(json);

        if (root == null || root.trajectory == null)
        {
            Debug.LogError("Invalid trajectory JSON.");
            return;
        }

        var traj = root.trajectory;
        waypoints.Clear();

        if (traj.x.Count > 0)
        {
            Vector3 rawFirstWaypoint = new Vector3(traj.x[0], traj.y[0], traj.z[0]);
            Vector3 offset = startOffsetPos - rawFirstWaypoint;

            // Capture the object's starting rotation to act as a baseline offset
            Quaternion initialRotation = transform.rotation;

            for (int i = 0; i < traj.x.Count; i++)
            {
                Vector3 rawPoint = new Vector3(traj.x[i], traj.y[i], traj.z[i]);
                Vector3 finalPos = rawPoint + offset;
                
                // --- UPDATED: Parse theta and convert from Radians to Degrees ---
                // Based on your data, x and z move drastically while y remains relatively flat, 
                // meaning theta represents Yaw (rotation around the Y-axis).
                float yawDegrees = 0f;
                if (traj.theta != null && traj.theta.Count > i)
                {
                    yawDegrees = traj.theta[i] * Mathf.Rad2Deg;
                }

                // Combine the JSON yaw rotation with your car's initial editor rotation
                Quaternion trajectoryRotation = initialRotation * Quaternion.Euler(0f, yawDegrees, 0f);
                
                waypoints.Add(new TrajectoryGenerator.TrajectoryPoint(finalPos, trajectoryRotation));
            }
        }
    }

    void Update()
    {
        if (currentWaypoint >= waypoints.Count)
            return;

        TrajectoryGenerator.TrajectoryPoint targetPoint = waypoints[currentWaypoint];

        // Move towards the target position
        transform.position = Vector3.MoveTowards(
            transform.position,
            targetPoint.position,
            speed * Time.deltaTime
        );

        // Handle Rotation
        if (useStoredRotation)
        {
            transform.rotation = Quaternion.RotateTowards(
                transform.rotation,
                targetPoint.rotation,
                turnSpeedDegrees * Time.deltaTime
            );
        }
        else
        {
            Vector3 direction = targetPoint.position - transform.position;
            if (direction.sqrMagnitude > 0.0001f)
            {
                Quaternion targetRotation = Quaternion.LookRotation(direction.normalized);
                transform.rotation = Quaternion.RotateTowards(
                    transform.rotation,
                    targetRotation,
                    turnSpeedDegrees * Time.deltaTime
                );
            }
        }

        // Advance to next waypoint
        if (Vector3.Distance(transform.position, targetPoint.position) < waypointThreshold)
        {
            currentWaypoint++;
        }
    }

    // --- UPDATED JSON Classes ---
    [Serializable]
    public class TrajectoryContainer 
    { 
        public List<float> x; 
        public List<float> y; 
        public List<float> z; 
        public List<float> theta; // Added to map the JSON's orientation array
    }
    
    [Serializable]
    public class RootTrajectoryData { public TrajectoryContainer trajectory; }
}