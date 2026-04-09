using UnityEngine;

public class FixedPositionRevealRotator : MonoBehaviour
{
    [SerializeField] private GameObject modelTemplate;
    [SerializeField] private Transform spawnParent;
    [SerializeField] private Vector3 defaultEulerAngles = Vector3.zero;
    [SerializeField] private Vector3 rotationEulerPerSecond = new Vector3(0f, 60f, 0f);

    private GameObject _latestInstance;

    public GameObject SpawnModel(Vector3 position)
    {
        return SpawnModel(position, Quaternion.Euler(defaultEulerAngles), null, false);
    }

    public GameObject SpawnModel(Vector3 position, Quaternion rotation)
    {
        return SpawnModel(position, rotation, null, false);
    }

    public GameObject SpawnModel(Vector3 position, Quaternion rotation, Color color)
    {
        return SpawnModel(position, rotation, color, false);
    }

    public GameObject SpawnModel(Vector3 position, Quaternion rotation, Color? color, bool rotateAfterSpawn)
    {
        if (modelTemplate == null)
        {
            Debug.LogError("[FixedPositionRevealRotator] Model template is not assigned.", this);
            return null;
        }

        GameObject instance = Instantiate(modelTemplate, position, rotation, spawnParent);
        instance.SetActive(true);

        if (color.HasValue)
        {
            ApplyColor(instance, color.Value);
        }

        if (rotateAfterSpawn)
        {
            EnsureRuntimeRotator(instance);
        }

        _latestInstance = instance;
        return instance;
    }

    public GameObject SpawnModelWithEuler(Vector3 position, Vector3 eulerAngles, Color? color, bool rotateAfterSpawn)
    {
        return SpawnModel(position, Quaternion.Euler(eulerAngles), color, rotateAfterSpawn);
    }

    public void DestroyLatestInstance()
    {
        if (_latestInstance != null)
        {
            Destroy(_latestInstance);
            _latestInstance = null;
        }
    }

    public void DestroyInstance(GameObject instance)
    {
        if (instance == null)
        {
            return;
        }

        if (_latestInstance == instance)
        {
            _latestInstance = null;
        }

        Destroy(instance);
    }

    void ApplyColor(GameObject target, Color color)
    {
        Renderer[] renderers = target.GetComponentsInChildren<Renderer>(true);
        foreach (Renderer rendererComponent in renderers)
        {
            Material[] materials = rendererComponent.materials;
            foreach (Material material in materials)
            {
                if (material.HasProperty("_BaseColor"))
                {
                    material.SetColor("_BaseColor", color);
                }
                else if (material.HasProperty("_Color"))
                {
                    material.SetColor("_Color", color);
                }
            }
        }
    }

    void EnsureRuntimeRotator(GameObject target)
    {
        RuntimeSpin spin = target.GetComponent<RuntimeSpin>();
        if (spin == null)
        {
            spin = target.AddComponent<RuntimeSpin>();
        }

        spin.SetAngularVelocity(rotationEulerPerSecond);
    }

    public class RuntimeSpin : MonoBehaviour
    {
        private Vector3 _angularVelocity;

        public void SetAngularVelocity(Vector3 angularVelocity)
        {
            _angularVelocity = angularVelocity;
        }

        void Update()
        {
            transform.Rotate(_angularVelocity * Time.deltaTime, Space.Self);
        }
    }
}
