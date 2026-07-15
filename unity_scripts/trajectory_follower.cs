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

    [Header("Rotation")]
    [Tooltip("Degrees per second the car can rotate.")]
    public float turnSpeedDegrees = 90f;

    [Tooltip("Distance to a waypoint before advancing to the next one.")]
    public float waypointThreshold = 0.3f;

    private List<Vector3> waypoints = new List<Vector3>();
    private int currentWaypoint = 0;

    void Start()
    {
        if (loadFromJson)
        {
            // --- SOURCE A: Load from JSON ---
            Vector3 startingPosition = transform.position;
            LoadTrajectoryFromJson(startingPosition);
        }
        else
        {
            // --- SOURCE B: Get from the Generator Script ---
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

        // Initialize movement if we have waypoints
        if (waypoints.Count > 0)
        {
            // If using the generator, the first waypoint is already exactly where the car starts,
            // so we start moving towards waypoint index 1.
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

            for (int i = 0; i < traj.x.Count; i++)
            {
                Vector3 rawPoint = new Vector3(traj.x[i], traj.y[i], traj.z[i]);
                waypoints.Add(rawPoint + offset);
            }
        }
    }

    void Update()
    {
        if (currentWaypoint >= waypoints.Count)
            return;

        Vector3 target = waypoints[currentWaypoint];

        // Move
        transform.position = Vector3.MoveTowards(
            transform.position,
            target,
            speed * Time.deltaTime
        );

        // Rotate
        Vector3 direction = target - transform.position;

        if (direction.sqrMagnitude > 0.0001f)
        {
            Quaternion targetRotation = Quaternion.LookRotation(direction.normalized);
            transform.rotation = Quaternion.RotateTowards(
                transform.rotation,
                targetRotation,
                turnSpeedDegrees * Time.deltaTime
            );
        }

        // Advance to next waypoint
        if (Vector3.Distance(transform.position, target) < waypointThreshold)
        {
            currentWaypoint++;
        }
    }

    // --- JSON Classes ---
    [Serializable]
    public class TrajectoryContainer { public List<float> x; public List<float> y; public List<float> z; }
    [Serializable]
    public class RootTrajectoryData { public TrajectoryContainer trajectory; }
}