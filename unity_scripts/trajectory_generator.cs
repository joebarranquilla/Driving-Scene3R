using System;
using System.Collections.Generic;
using UnityEngine;

public class TrajectoryGenerator : MonoBehaviour
{
    [Header("Destination")]
    [Tooltip("The final XYZ coordinates you want the object to reach.")]
    public Vector3 targetPosition;

    [Header("Path Settings")]
    [Tooltip("How many waypoints to generate. Higher numbers mean a smoother curve.")]
    [Range(5, 100)]
    public int resolution = 30;

    [Tooltip("Controls the curve shape. Pulls the middle of the path toward this offset.")]
    public Vector3 curveOffset = new Vector3(0, 5f, 0);

    [Header("Gizmos (Editor Preview)")]
    public bool showPreview = true;
    public Color previewColor = Color.green;

    // This list holds the generated path
    private List<Vector3> generatedWaypoints = new List<Vector3>();

    void Start()
    {
        GeneratePath();
    }

    /// <summary>
    /// Generates a smooth Bezier curve from the object's current position to the target position.
    /// </summary>
    public List<Vector3> GeneratePath()
    {
        generatedWaypoints.Clear();

        Vector3 startPoint = transform.position;
        Vector3 endPoint = targetPosition;

        // We create a control point in the middle to give the path a smooth curve.
        // It's halfway between start and end, shifted by our custom curveOffset.
        Vector3 controlPoint = Vector3.Lerp(startPoint, endPoint, 0.5f) + curveOffset;

        for (int i = 0; i < resolution; i++)
        {
            // t goes from 0.0 (start) to 1.0 (end)
            float t = i / (float)(resolution - 1);
            
            // Calculate Quadratic Bezier Curve point
            Vector3 pathPoint = CalculateBezierPoint(t, startPoint, controlPoint, endPoint);
            generatedWaypoints.Add(pathPoint);
        }

        Debug.Log($"Generated a trajectory with {generatedWaypoints.Count} points starting from {startPoint} to {endPoint}.");
        return generatedWaypoints;
    }

    /// <summary>
    /// Quadratic Bezier formula: B(t) = (1-t)^2 * P0 + 2(1-t)t * P1 + t^2 * P2
    /// </summary>
    private Vector3 CalculateBezierPoint(float t, Vector3 p0, Vector3 p1, Vector3 p2)
    {
        float u = 1 - t;
        float tt = t * t;
        float uu = u * u;

        Vector3 point = uu * p0; // (1-t)^2 * P0
        point += 2 * u * t * p1; // 2(1-t)t * P1
        point += tt * p2;        // t^2 * P2

        return point;
    }

    // Public getter so your TrajectoryFollower script can access these points
    public List<Vector3> GetWaypoints()
    {
        if (generatedWaypoints.Count == 0)
        {
            GeneratePath();
        }
        return generatedWaypoints;
    }

    // Draws a line in the Unity Scene view so you can see the path before pressing Play
    private void OnDrawGizmos()
    {
        if (!showPreview) return;

        Vector3 startPoint = transform.position;
        Vector3 endPoint = targetPosition;
        Vector3 controlPoint = Vector3.Lerp(startPoint, endPoint, 0.5f) + curveOffset;

        Gizmos.color = previewColor;
        Vector3 previousPoint = startPoint;

        for (int i = 1; i <= resolution; i++)
        {
            float t = i / (float)resolution;
            Vector3 currentPoint = CalculateBezierPoint(t, startPoint, controlPoint, endPoint);
            Gizmos.DrawLine(previousPoint, currentPoint);
            previousPoint = currentPoint;
        }

        // Draw helper spheres at the start, control, and end points
        Gizmos.color = Color.blue;
        Gizmos.DrawWireSphere(startPoint, 0.3f);
        Gizmos.color = Color.red;
        Gizmos.DrawWireSphere(endPoint, 0.3f);
    }
}