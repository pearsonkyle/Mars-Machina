using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using TensorFlow;
public class meshGenerator : MonoBehaviour
{
    // stuff for creating a simple mesh
    Material material;
    Mesh mesh;
    Vector3[] vertices;
    int[] triangles;
    Vector2[] uvs;

    // output size from NN float[784] -> float[28,28]
    public int xSize = 96;
    public int ySize = 96;


    // input to machine learning model 
    public TextAsset graphModel;
    TFGraph graph;
    TFSession session;

    private float[,] inputData =
        new float[1, 2];
    public float latent_x;
    public float latent_y;
    public float latent_speed = 1f;

    void Start()
    {
        
        mesh = new Mesh();
        GetComponent<MeshFilter>().mesh = mesh;
        material = GetComponent<Renderer>().material;

        // Create the TensorFlow model
        graph = new TFGraph();
        graph.Import(graphModel.bytes);
        session = new TFSession(graph);

        // start generating at a random point in latent parameter space
        latent_x = Random.value;
        latent_y = Random.value;
    }


    void CreateShape()
    {
        // Run the model
        var runner = session.GetRunner();
        runner.AddInput(graph["decoder_input"][0], inputData);
        runner.Fetch(graph["decoder_output/Sigmoid"][0]);
        float[,,,] newImage = runner.Run()[0].GetValue() as float[,,,];

        // Debug.Log("new image:"+newImage.Length);

        vertices = new Vector3[(xSize + 1) * (ySize + 1)];

        int i = 0;
        for (int y = 0; y < ySize + 1; y++)
        {
            for (int x = 0; x < xSize + 1; x++)
            {
                vertices[i] = new Vector3( (float)x/xSize, (float)y/ySize, newImage[0, Mathf.Clamp(x, 0, xSize-1), Mathf.Clamp(y,0,ySize-1), 0] );

                i++;
            }
        }

        triangles = new int[xSize * ySize * 6];

        int vert = 0;
        int tris = 0;
        for (int y = 0; y < ySize; y++)
        {
            for (int x = 0; x < xSize; x++)
            {

                triangles[tris + 0] = vert + 0;
                triangles[tris + 1] = vert + xSize + 1;
                triangles[tris + 2] = vert + 1;
                triangles[tris + 3] = vert + 1;
                triangles[tris + 4] = vert + xSize + 1;
                triangles[tris + 5] = vert + xSize + 2;

                vert++;
                tris += 6;
            }
            vert++;
        }
    }

    
    void UpdateMesh()
    {
        mesh.Clear();
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
    }
    private void OnDrawGizmos()
    {
        if (vertices == null)
            return;

        for (int i = 0; i < vertices.Length; i++)
        {
            Gizmos.DrawSphere(vertices[i], 0.1f);
        }
    }


    Touch touch;
    float lastTime=0;
    // Update is called once per frame
    void Update()
    {
        CreateShape();
        UpdateMesh();

        // touch controls
        if (Input.touchCount == 1)
        {
            // rotate player based on position derivative
            if (Input.GetTouch(0).phase == TouchPhase.Moved)
            {
                touch = Input.GetTouch(0);
                latent_x = 4*(touch.position.x - Screen.width/2)/Screen.width;
                latent_y = 4*(touch.position.y - Screen.height/2) / Screen.height;

                inputData[0, 0] = latent_x;
                inputData[0, 1] = latent_y;
                lastTime = Time.time;
            }
        }
        else
        {
            // input data should only vary between ~-4 -> 4
            inputData[0, 0] = latent_x + 2 * Mathf.Sin(latent_speed* 0.2f * (Time.time-lastTime));
            inputData[0, 1] = latent_y + 2 * Mathf.Sin(1.5f + 0.25f * latent_speed * (Time.time-lastTime));
        }
        // Color of texture will change depending on ange of reflected light
        //Color newColor = new Vector4(inputData[0,0], inputData[0, 1]*0.75f, inputData[0, 1], 1);
        //material.SetColor("_ColorDirect1", newColor);
    }
}
