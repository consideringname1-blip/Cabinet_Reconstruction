using System;
using System.Collections.Generic;
using UnityEngine;

public class RuntimeSkeletonJointData
{
    public string Name = "";
    public Vector3 PositionWorld = Vector3.zero;
    public float Score;
    public bool Valid;
}

public class RuntimeSkeletonData
{
    public string PeopleId = "";
    public readonly List<RuntimeSkeletonJointData> Joints =
        new List<RuntimeSkeletonJointData>();
}

[DisallowMultipleComponent]
public class RuntimeSkeletonDisplay : MonoBehaviour
{
    private const float JointDiameterMeters = 0.018f;
    private const float BoneDiameterMeters = 0.008f;

    private static readonly string[,] BonePairs =
    {
        { "nose", "neck" },
        { "neck", "right_shoulder" },
        { "right_shoulder", "right_elbow" },
        { "right_elbow", "right_wrist" },
        { "neck", "left_shoulder" },
        { "left_shoulder", "left_elbow" },
        { "left_elbow", "left_wrist" },
        { "neck", "mid_hip" },
        { "mid_hip", "right_hip" },
        { "right_hip", "right_knee" },
        { "right_knee", "right_ankle" },
        { "mid_hip", "left_hip" },
        { "left_hip", "left_knee" },
        { "left_knee", "left_ankle" },
        { "nose", "right_eye" },
        { "right_eye", "right_ear" },
        { "nose", "left_eye" },
        { "left_eye", "left_ear" },
        { "left_ankle", "left_big_toe" },
        { "left_big_toe", "left_small_toe" },
        { "left_ankle", "left_heel" },
        { "right_ankle", "right_big_toe" },
        { "right_big_toe", "right_small_toe" },
        { "right_ankle", "right_heel" },
    };

    private Material leftMaterial;
    private Material rightMaterial;
    private Material torsoMaterial;

    public void Configure(RuntimeSkeletonData skeleton)
    {
        ClearGeometry();
        if (skeleton == null || skeleton.Joints == null)
        {
            return;
        }

        EnsureMaterials();
        Dictionary<string, RuntimeSkeletonJointData> joints =
            new Dictionary<string, RuntimeSkeletonJointData>(StringComparer.Ordinal);
        foreach (RuntimeSkeletonJointData joint in skeleton.Joints)
        {
            if (!IsRenderable(joint) || joints.ContainsKey(joint.Name))
            {
                continue;
            }
            joints[joint.Name] = joint;
            CreateJoint(joint);
        }

        for (int i = 0; i < BonePairs.GetLength(0); i++)
        {
            string startName = BonePairs[i, 0];
            string endName = BonePairs[i, 1];
            if (joints.TryGetValue(startName, out RuntimeSkeletonJointData start)
                && joints.TryGetValue(endName, out RuntimeSkeletonJointData end))
            {
                CreateBone(i, start, end);
            }
        }
    }

    private void OnDestroy()
    {
        DestroyMaterial(leftMaterial);
        DestroyMaterial(rightMaterial);
        DestroyMaterial(torsoMaterial);
    }

    private void ClearGeometry()
    {
        for (int i = transform.childCount - 1; i >= 0; i--)
        {
            Destroy(transform.GetChild(i).gameObject);
        }
    }

    private void CreateJoint(RuntimeSkeletonJointData joint)
    {
        GameObject point = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        point.name = "skeleton_joint_" + joint.Name;
        point.transform.SetParent(transform, true);
        point.transform.position = joint.PositionWorld;
        point.transform.localScale = Vector3.one * JointDiameterMeters;
        RemoveCollider(point);
        SetMaterial(point, MaterialForJoint(joint.Name));
    }

    private void CreateBone(
        int index,
        RuntimeSkeletonJointData start,
        RuntimeSkeletonJointData end)
    {
        Vector3 delta = end.PositionWorld - start.PositionWorld;
        float length = delta.magnitude;
        if (length <= 0.000001f)
        {
            return;
        }

        GameObject bone = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        bone.name = "skeleton_bone_" + index.ToString();
        bone.transform.SetParent(transform, true);
        bone.transform.position = (start.PositionWorld + end.PositionWorld) * 0.5f;
        bone.transform.rotation = Quaternion.FromToRotation(
            Vector3.up,
            delta / length);
        // Unity's primitive cylinder has radius 0.5 and height 2.
        bone.transform.localScale = new Vector3(
            BoneDiameterMeters,
            length * 0.5f,
            BoneDiameterMeters);
        RemoveCollider(bone);
        SetMaterial(bone, MaterialForBone(start.Name, end.Name));
    }

    private void EnsureMaterials()
    {
        if (leftMaterial == null)
        {
            leftMaterial = CreateMaterial(new Color(0.0f, 1.0f, 1.0f, 1.0f));
        }
        if (rightMaterial == null)
        {
            rightMaterial = CreateMaterial(new Color(1.0f, 0.0f, 0.85f, 1.0f));
        }
        if (torsoMaterial == null)
        {
            torsoMaterial = CreateMaterial(new Color(1.0f, 0.85f, 0.0f, 1.0f));
        }
    }

    private Material MaterialForJoint(string jointName)
    {
        if (IsLeft(jointName))
        {
            return leftMaterial;
        }
        if (IsRight(jointName))
        {
            return rightMaterial;
        }
        return torsoMaterial;
    }

    private Material MaterialForBone(string startName, string endName)
    {
        if (IsLeft(startName) || IsLeft(endName))
        {
            return leftMaterial;
        }
        if (IsRight(startName) || IsRight(endName))
        {
            return rightMaterial;
        }
        return torsoMaterial;
    }

    private static bool IsLeft(string name)
    {
        return !string.IsNullOrEmpty(name)
            && name.StartsWith("left_", StringComparison.Ordinal);
    }

    private static bool IsRight(string name)
    {
        return !string.IsNullOrEmpty(name)
            && name.StartsWith("right_", StringComparison.Ordinal);
    }

    private static bool IsRenderable(RuntimeSkeletonJointData joint)
    {
        return joint != null
            && joint.Valid
            && !string.IsNullOrEmpty(joint.Name)
            && IsFinite(joint.PositionWorld);
    }

    private static bool IsFinite(Vector3 value)
    {
        return !float.IsNaN(value.x)
            && !float.IsInfinity(value.x)
            && !float.IsNaN(value.y)
            && !float.IsInfinity(value.y)
            && !float.IsNaN(value.z)
            && !float.IsInfinity(value.z);
    }

    private static Material CreateMaterial(Color color)
    {
        Shader shader = Shader.Find("Unlit/Color");
        if (shader == null)
        {
            shader = Shader.Find("Standard");
        }
        if (shader == null)
        {
            shader = Shader.Find("Hidden/InternalErrorShader");
        }
        Material material = new Material(shader);
        if (material.HasProperty("_Color"))
        {
            material.SetColor("_Color", color);
        }
        if (material.HasProperty("_BaseColor"))
        {
            material.SetColor("_BaseColor", color);
        }
        return material;
    }

    private static void RemoveCollider(GameObject target)
    {
        Collider collider = target != null ? target.GetComponent<Collider>() : null;
        if (collider != null)
        {
            Destroy(collider);
        }
    }

    private static void SetMaterial(GameObject target, Material material)
    {
        Renderer renderer = target != null ? target.GetComponent<Renderer>() : null;
        if (renderer != null)
        {
            renderer.sharedMaterial = material;
        }
    }

    private static void DestroyMaterial(Material material)
    {
        if (material != null)
        {
            Destroy(material);
        }
    }
}
