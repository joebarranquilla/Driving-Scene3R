using System;
using System.Collections.Generic;
using UnityEngine;

public class TrajectoryGenerator : MonoBehaviour
{
    // A custom struct to bundle position and rotation together
    [Serializable]
    public struct TrajectoryPoint
    {
        public Vector3 position;
        public Quaternion rotation;

        public TrajectoryPoint(Vector3 position, Quaternion rotation)
        {
            this.position = position;
            this.rotation = rotation;
        }
    }

    [Header("Destination")]
    [Tooltip("The final XYZ coordinates you want the object to reach.")]
    public Vector3 targetPosition;
    
    [Tooltip("The final rotation (in Euler angles) you want the object to have at the destination.")]
    public Vector3 targetRotationEuler;

    [Header("Path Settings")]
    [Tooltip("How many waypoints to generate. Higher numbers mean a smoother curve.")]
    [Range(5, 100)]
    public int resolution = 30;

    [Tooltip("Controls the curve shape. Pulls the middle of the path toward this offset.")]
    public Vector3 curveOffset = new Vector3(0, 5f, 0);

    [Tooltip("If true, rotates smoothly from start to target rotation. If false, strictly keeps the object's original starting rotation across the whole path.")]
    public bool interpolateRotation = false;

    [Header("Gizmos (Editor Preview)")]
    public bool showPreview = true;
    public Color previewColor = Color.green;

    // Modified to hold our new struct containing both position and rotation
    private List<TrajectoryPoint> generatedWaypoints = new List<TrajectoryPoint>();

    void Start()
    {
        GeneratePath();
    }

    /// <summary>
    /// Generates a smooth Bezier curve keeping or interpolating the rotation data.
    /// </summary>
    public List<TrajectoryPoint> GeneratePath()
    {
        generatedWaypoints.Clear();

        Vector3 startPoint = transform.position;
        Vector3 endPoint = targetPosition;

        // Capture the original rotation degrees of the object at the moment of generation
        Quaternion startRotation = transform.rotation;
        Quaternion endRotation = interpolateRotation ? Quaternion.Euler(targetRotationEuler) : startRotation;

        Vector3 controlPoint = Vector3.Lerp(startPoint, endPoint, 0.5f) + curveOffset;

        for (int i = 0; i < resolution; i++)
        {
            float t = i / (float)(resolution - 1);
            
            // Calculate Position
            Vector3 pathPoint = CalculateBezierPoint(t, startPoint, controlPoint, endPoint);
            
            // Calculate Rotation (Slerp handles spherical interpolation seamlessly)
            Quaternion pathRotation = Quaternion.Slerp(startRotation, endRotation, t);

            generatedWaypoints.Add(new TrajectoryPoint(pathPoint, pathRotation));
        }

        Debug.Log($"Generated a trajectory with {generatedWaypoints.Count} points. Rotation tracking active.");
        return generatedWaypoints;
    }

    private Vector3 CalculateBezierPoint(float t, Vector3 p0, Vector3 p1, Vector3 p2)
    {
        float u = 1 - t;
        float tt = t * t;
        float uu = u * u;

        Vector3 point = uu * p0;
        point += 2 * u * t * p1;
        point += tt * p2;

        return point;
    }

    // Updated public getter returning the combined dataset
    public List<TrajectoryPoint> GetWaypoints()
    {
        if (generatedWaypoints.Count == 0)
        {
            GeneratePath();
        }
        return generatedWaypoints;
    }

    private void OnDrawGizmos()
    {
        if (!showPreview) return;

        Vector3 startPoint = transform.position;
        Vector3 endPoint = targetPosition;
        Vector3 controlPoint = Vector3.Lerp(startPoint, endPoint, 0.5f) + curveOffset;

        Gizmos.color = previewColor;
        Vector3 previousPosition = startPoint;

        Quaternion startRotation = transform.rotation;
        Quaternion endRotation = interpolateRotation ? Quaternion.Euler(targetRotationEuler) : startRotation;

        for (int i = 1; i <= resolution; i++)
        {
            float t = i / (float)resolution;
            Vector3 currentPosition = CalculateBezierPoint(t, startPoint, controlPoint, endPoint);
            Gizmos.DrawLine(previousPosition, currentPosition);

            // Draw small orientation axes along the path preview
            Quaternion currentRotation = Quaternion.Slerp(startRotation, endRotation, t);
            Gizmos.color = Color.red; // Forward axis indicator
            Gizmos.DrawRay(currentPosition, currentRotation * Vector3.forward * 0.4f);

            previousPosition = currentPosition;
        }

        Gizmos.color = Color.blue;
        Gizmos.DrawWireSphere(startPoint, 0.3f);
        Gizmos.color = Color.red;
        Gizmos.DrawWireSphere(endPoint, 0.3f);
    }
}