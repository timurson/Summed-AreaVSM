#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "glsw.h"
#include "model.h"
#include "shader_s.h"
#include "arcball_camera.h"
#include "framebuffer.h"
#include "utility.h"
#include "openglblurdata.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
#include <GL/gl3w.h>            // Initialize with gl3wInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
#include <GL/glew.h>            // Initialize with glewInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
#include <glad/glad.h>          // Initialize with gladLoadGL()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING2)
#define GLFW_INCLUDE_NONE       // GLFW including OpenGL headers causes ambiguity or multiple definition errors.
#include <glbinding/Binding.h>  // Initialize with glbinding::Binding::initialize()
#include <glbinding/gl/gl.h>
using namespace gl;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING3)
#define GLFW_INCLUDE_NONE       // GLFW including OpenGL headers causes ambiguity or multiple definition errors.
#include <glbinding/glbinding.h>// Initialize with glbinding::initialize()
#include <glbinding/gl/gl.h>
using namespace gl;
#else
#include IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#endif

#include <iostream>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

#define PATH fs::current_path().generic_string()

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
unsigned int loadTexture(const char *path, bool gammaCorrection);
void renderQuad();
void renderCube();
void renderCubemap(int cubemap, Shader& equirectangularToCubemapShader, Shader& irradianceShader);


// settings
const unsigned int SCR_WIDTH = 1024;
const unsigned int SCR_HEIGHT = 768;
//const unsigned int SCR_WIDTH = 2048;
//const unsigned int SCR_HEIGHT = 1535;
const unsigned int SHADOW_MAP_SIZE = 1024;
const unsigned int ENV_CUBEMAP_SIZE = 512;
const unsigned int IRRADIANCE_CUBEMAP_SIZE = 64;
const float MAX_CAMERA_DISTANCE = 200.0f;
const unsigned int LIGHT_GRID_WIDTH = 5;  // point light grid size
const unsigned int LIGHT_GRID_HEIGHT = 4;  // point light vertical grid height
const float INITIAL_POINT_LIGHT_RADIUS = 0.870f;

// compute shader related:
// 16 and 32 do well on BYT, anything in between or below is bad, values above were not thoroughly tested; 32 seems to do well on laptop/desktop Windows Intel and on NVidia/AMD as well
// (further hardware-specific tuning probably needed for optimal performance)
static const int CS_THREAD_GROUP_SIZE = 32;


// camera
ArcballCamera arcballCamera(glm::vec3(0.0f, 1.5f, 5.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
// global light
ArcballCamera arcballLight(glm::vec3(-2.5f, 5.0f, -1.25f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

float lastX = (float)SCR_WIDTH / 2.0;
float lastY = (float)SCR_HEIGHT / 2.0;
bool firstMouse = true;
bool leftMouseButtonPressed = false;
bool rightMouseButtonPressed = false;
int mouseControl = 0;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

enum GBufferRender : int8_t
{
    Final,
    WorldPosition,
    WorldNormal,
    Diffuse,
    Specular,
    Occlusion, // ambient occlusion
    Count
};

// struct to hold information about scene light
struct SceneLight {
    SceneLight(const glm::vec3& _position, const glm::vec3& _color, float _radius, float _intensity)
        : position(_position), color(_color), radius(_radius), intensity(_intensity)
    {}
    glm::vec3 position;      // world light position
    glm::vec3 color;         // light's color
    float     radius;        // light's radius
    float     intensity;     // light's intensity

};

struct Material {
    Material(const glm::vec3& _diffuse, const glm::vec3& _specular, float _roughness, float _metallic)
        :diffuse(_diffuse), specular(_specular), roughness(_roughness), metallic(_metallic)
    {}
    glm::vec3 diffuse;      // material diffuse color
    glm::vec3 specular;     // material specular color
    float     roughness;    // material roughness 
    float     metallic;     // how metalic material is
};

// buffer for light instance data
unsigned int matrixBuffer;
unsigned int colorSizeBuffer;

// cubemap and irradiance map ids
unsigned int envCubemap = 0;
unsigned int irradianceMap = 0;
unsigned int hdrTexture = 0;
// fbo for rendering into cubemap and LUT texture
unsigned int captureFBO;
unsigned int captureRBO;

void configurePointLights(std::vector<glm::mat4>& modelMatrices, std::vector<glm::vec4>& modelColorSizes, float radius = 1.0f, float separation = 1.0f, float yOffset = 0.0f);
void updatePointLights(std::vector<glm::mat4>& modelMatrices, std::vector<glm::vec4>& modelColorSizes, float separation, float yOffset, float radiusScale);

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    const char* glsl_version = "#version 430";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Summed-Area Soft Variance Shadows (Roman Timurson)", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // tell stb_image.h to flip loaded texture's on the y-axis (before loading model).
    stbi_set_flip_vertically_on_load(true);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // configure global opengl state
    // -----------------------------
    glEnable(GL_DEPTH_TEST);
    // set depth function to less than AND equal for skybox depth trick.
    glDepthFunc(GL_LEQUAL);
    // enable seamless cubemap sampling for lower mip levels in the pre-filter map.
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

    std::string globalShaderConstants;
    //int computeShaderKernelSize = 15; // 7, 15, 23, 35, 63, 127
    int computeShaderKernel[6]{ 7, 15, 23, 35, 63, 127 };

    glswInit();
    glswSetPath("OpenGL/shaders/", ".glsl");
    glswAddDirectiveToken("", "#version 430 core");

    // define shader constants
    globalShaderConstants = cStringFormatA("#define cRTScreenSizeI ivec4( %d, %d, %d, %d ) \n", SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
    glswAddDirectiveToken("*", globalShaderConstants.c_str());

    //globalShaderConstants = cStringFormatA("#define COMPUTE_SHADER_KERNEL_SIZE %d\n", computeShaderKernelSize);
    //glswAddDirectiveToken("*", globalShaderConstants.c_str());

    globalShaderConstants = cStringFormatA("#define CS_THREAD_GROUP_SIZE %d\n", CS_THREAD_GROUP_SIZE);
    glswAddDirectiveToken("*", globalShaderConstants.c_str());


    // SAT
    Shader shaderSATHorizontal(glswGetShader("SAT.Vertex"), glswGetShader("SAT.FragmentH"));
    Shader shaderSATVertical(glswGetShader("SAT.Vertex"), glswGetShader("SAT.FragmentV"));
    Shader computeSAT(glswGetShader("computeSAT.ComputeSAT"));
    // hdr cubemap shaders
    Shader equirectangularToCubemapShader(glswGetShader("equirectToCubemap.Vertex"), glswGetShader("equirectToCubemap.Fragment"));
    Shader cubemapShader(glswGetShader("cubemap.Vertex"), glswGetShader("cubemap.Fragment"));
    // SSAO shaders
    Shader shaderSSAO(glswGetShader("ambientOcclusion.Vertex"), glswGetShader("ambientOcclusion.Fragment"));
    Shader computeBilateralBlur(glswGetShader("bilateralBlur.Compute"));
    // PBR irradiance generation shader
    Shader irradianceShader(glswGetShader("irradianceGen.Vertex"), glswGetShader("irradianceGen.Fragment"));
    // BRDR LUT generation shader
    Shader brdfShader(glswGetShader("brdf.Vertex"), glswGetShader("brdf.Fragment"));
    // Shader for writing into a depth texture
    Shader shaderDepthWrite(glswGetShader("varianceShadowMap.Vertex"), glswGetShader("varianceShadowMap.Fragment"));
    // Compute shader for doing multi-pass moving average box filtering
    Shader computeBlurShaderH(glswGetShader("blurCompute.ComputeH"));
    Shader computeBlurShaderV(glswGetShader("blurCompute.ComputeV"));
    // Shader for visualizing the depth texture
    Shader shaderDebugDepthMap(glswGetShader("debugMSM.Vertex"), glswGetShader("debugMSM.Fragment"));
    // Shader for visualizing cubemaps as equirectangular textures
    Shader shaderDebugCubemap(glswGetShader("debugCubemap.Vertex"), glswGetShader("debugCubemap.Fragment"));
    // G-Buffer pass shader for models w/o textures and just Kd, Ks, etc colors 
    Shader shaderGeometryPass(glswGetShader("gBuffer.Vertex"), glswGetShader("gBuffer.Fragment"));
    // G-Buffer pass shader for the models with textures (diffuse, specular, etc)
    Shader shaderTexturedGeometryPass(glswGetShader("gBufferTextured.Vertex"), glswGetShader("gBufferTextured.Fragment"));
    // First pass of deferred PBR shader that will render the scene with a global light and shadow mapping
    Shader pbrShader(glswGetShader("deferredSASVSM.Vertex"), glswGetShader("deferredSASVSM.Fragment"));
    // Shader for debugging the G-Buffer contents
    Shader shaderGBufferDebug(glswGetShader("gBufferDebug.Vertex"), glswGetShader("gBufferDebug.Fragment"));
    // Shader for debugging ambient occlusion map
    Shader shaderSSAODebug(glswGetShader("ssaoDebug.Vertex"), glswGetShader("ssaoDebug.Fragment"));
    // Shader to render the light geometry for visualization and debugging
    Shader shaderGlobalLightSphere(glswGetShader("deferredLight.Vertex"), glswGetShader("deferredLight.Fragment"));
    Shader shaderLightSphere(glswGetShader("deferredLightInstanced.Vertex"), glswGetShader("deferredLightInstanced.Fragment"));
    // Shader for a final composite rendering of point(area) lights with generated G-Buffer
    Shader shaderPointLightingPass(glswGetShader("deferredPointLightInstanced.Vertex"), glswGetShader("deferredPointLightInstanced.Fragment"));

    // pbr: load the HDR environment map and render it into cubemap
    // ---------------------------------
    glGenFramebuffers(1, &captureFBO);
    glGenRenderbuffers(1, &captureRBO);

    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, ENV_CUBEMAP_SIZE, ENV_CUBEMAP_SIZE);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, captureRBO);

    // pbr: generate an environment and irradiance cubemaps for IBL
    renderCubemap(0, equirectangularToCubemapShader, irradianceShader);

    // pbr: generate a 2D LUT from the BRDF equations used.
    // ----------------------------------------------------
    unsigned int brdfLUTTexture;
    glGenTextures(1, &brdfLUTTexture);
    // pre-allocate enough memory for the LUT texture.
    glBindTexture(GL_TEXTURE_2D, brdfLUTTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, 512, 512, 0, GL_RG, GL_FLOAT, 0);

    // be sure to set wrapping mode to GL_CLAMP_TO_EDGE
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // then re-configure capture framebuffer object and render screen-space quad with BRDF shader.
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 512, 512);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdfLUTTexture, 0);

    glViewport(0, 0, 512, 512);
    brdfShader.use();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    renderQuad();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    const float PLANE_HALF_WIDTH = 6.0f;
    float planeVertices[] = {
        // positions            // normals         // texcoords
         PLANE_HALF_WIDTH, -0.5f,  PLANE_HALF_WIDTH,  0.0f, 1.0f, 0.0f,  10.0f,  10.0f,
        -PLANE_HALF_WIDTH, -0.5f, -PLANE_HALF_WIDTH,  0.0f, 1.0f, 0.0f,   0.0f, 0.0f,
        -PLANE_HALF_WIDTH, -0.5f,  PLANE_HALF_WIDTH,  0.0f, 1.0f, 0.0f,   0.0f,  10.0f,
        
         PLANE_HALF_WIDTH, -0.5f,  PLANE_HALF_WIDTH,  0.0f, 1.0f, 0.0f,  10.0f,  10.0f,
         PLANE_HALF_WIDTH, -0.5f, -PLANE_HALF_WIDTH,  0.0f, 1.0f, 0.0f,  10.0f, 0.0f,
        -PLANE_HALF_WIDTH, -0.5f, -PLANE_HALF_WIDTH,  0.0f, 1.0f, 0.0f,   0.0f, 0.0f,
    };
    // create floor plane VAO
    unsigned int planeVAO, planeVBO;
    glGenVertexArrays(1, &planeVAO);
    glGenBuffers(1, &planeVBO);
    glBindVertexArray(planeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(planeVertices), planeVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glBindVertexArray(0);

    // load textures
    // -------------
    std::string woodTexturePath = PATH + "/OpenGL/images/wood.png";
    unsigned int woodTexture = loadTexture(woodTexturePath.c_str(), false);

    // load models
    // -----------
    std::string bunnyPath = PATH + "/OpenGL/models/Bunny.obj";
    std::string dragonPath = PATH + "/OpenGL/models/Dragon.obj";
    std::string ajaxPath = PATH + "/OpenGL/models/Ajax.obj";
    std::string lucyPath = PATH + "/OpenGL/models/Lucy.obj";
    std::string heptoroid = PATH + "/OpenGL/models/heptoroid.obj";
    //std::string modelPath = PATH + "/OpenGL/models/Aphrodite.obj";
    Model meshModelA(dragonPath);
    //Model meshModelB(dragonPath);
   // Model meshModelC(bunnyPath);
    std::string spherePath = PATH + "/OpenGL/models/Sphere.obj";
    Model lightModel(spherePath);
    
    std::vector<glm::vec3> objectPositions;
    objectPositions.push_back(glm::vec3(0.0, 0.4, 0.0));
    /*objectPositions.push_back(glm::vec3(-1.0, -0.5, 0.0));
    objectPositions.push_back(glm::vec3(0.0, -0.5, 0.0));
    objectPositions.push_back(glm::vec3(1.0, -0.5, 0.0));*/
   
    /* objectPositions.push_back(glm::vec3(2.5, 1.0, -0.5));
    objectPositions.push_back(glm::vec3(-2.5, 1.0, -0.5));
    objectPositions.push_back(glm::vec3(0.0, 1.0, 2.0));*/
    std::vector<Model*> meshModels;
    meshModels.push_back(&meshModelA);
   // meshModels.push_back(&meshModelA);
    //meshModels.push_back(&meshModelA);
   // meshModels.push_back(&meshModelB);
    //meshModels.push_back(&meshModelC);

    // configure depth map framebuffer for shadow generation/filtering
    // ----------------------
    FrameBuffer sBuffer(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
    sBuffer.attachTexture(GL_RGBA32F);
    sBuffer.attachTexture(GL_RGBA32F);            // attach secondary texture for ping-pong blurring
    sBuffer.attachRender(GL_DEPTH_COMPONENT32);     // attach Depth render buffer
    sBuffer.bindInput(0);
    // Remove artifacts on the edges of the shadowmap
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    float borderColor[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    sBuffer.bindInput(1);
    // Remove artifacts on the edges of the shadowmap
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    // configure SAT generation framebuffer
    FrameBuffer satBuffer(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
    satBuffer.attachTexture(GL_RGBA32F);
    satBuffer.attachTexture(GL_RGBA32F);
    satBuffer.attachRender(GL_DEPTH_COMPONENT32);     // attach Depth render buffer
    satBuffer.bindInput(0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);


    // configure g-buffer framebuffer
    // ------------------------------
    FrameBuffer gBuffer(SCR_WIDTH, SCR_HEIGHT);
    gBuffer.attachTexture(GL_RGBA16F, GL_LINEAR_MIPMAP_LINEAR); // Position color buffer + Depth
    gBuffer.attachTexture(GL_RGB16F, GL_NEAREST);  // Normal color buffer
    gBuffer.attachTexture(GL_RGBA, GL_NEAREST);    // Diffuse (Kd)
    gBuffer.attachTexture(GL_RGBA, GL_NEAREST);    // Specular (Ks)
    gBuffer.bindOutput();                          // calls glDrawBuffers[i] for all attached textures
    gBuffer.attachRender(GL_DEPTH_COMPONENT);      // attach Depth render buffer

    gBuffer.bindInput(0);
    glGenerateMipmap(GL_TEXTURE_2D);
    gBuffer.check();
    FrameBuffer::unbind();                        // unbind framebuffer for now

    // configure SSAO capture framebuffer
    FrameBuffer aoBuffer(SCR_WIDTH, SCR_HEIGHT);
    aoBuffer.attachTexture(GL_RGBA32F, GL_NEAREST);
    aoBuffer.attachTexture(GL_RGBA32F, GL_NEAREST);
    aoBuffer.bindOutput();                         // calls glDrawBuffers[i] for all attached textures
    aoBuffer.check();
    FrameBuffer::unbind();                         // unbind framebuffer for now

    // lighting info
    // -------------
    // instance array data for our light volumes
    std::vector<glm::mat4> modelMatrices;
    std::vector<glm::vec4> modelColorSizes;

    // single global light
    SceneLight globalLight(glm::vec3(-2.5f, 5.0f, -1.25f), glm::vec3(1.0f, 1.0f, 1.0f), 0.125f, 1.0f);

    // option settings
    int gBufferMode = 0;
    int KernelSizeOption = 0; // 7, 15, 23, 35, 63, 127
    int CubemapSelection = 0;
    bool enableShadows = true;
    bool drawPointLights = false;
    bool showDepthMap = false;
    bool drawPointLightsWireframe = true;
    // configure materials
    std::vector<Material> materials;
    //materials.push_back(Material(glm::vec3(98.0/255.0, 56.0/255.0, 30.0/255.0), glm::vec3(245.0/255.0, 229.0/255.0, 229.0/255.0), 0.715f, 0.0f));
    materials.push_back(Material(glm::vec3(0.0 / 255.0, 0.0 / 255.0, 0.0 / 255.0), glm::vec3(196.0 / 255.0, 172.0 / 255.0, 61.0 / 255.0), 0.2f, 1.0f));
    //materials.push_back(Material(glm::vec3(211.0 / 255.0, 186.0 / 255.0, 161.0 / 255.0), glm::vec3(255.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0), 0.226f, 0.072f));
    //float roughness = 0.2f;
    float pointLightIntensity = 0.545f;
    float pointLightRadius = INITIAL_POINT_LIGHT_RADIUS;
    float pointLightVerticalOffset = 1.205f;
    float pointLightSeparation = 0.620f;
    float shadowSaturation = 0.5f;
    float penumbraSize = 1.0f;
    int lightSourceRadius = 16;
    float modelScale = 0.9f;
    bool softSATVSM = false;
    // IBL
    int iblSamples = 30;
    // SSAO
    int aoSamples = 20;
    float sampleRadius = 1.0;
    float shadowScalar = 0.299;   // sigma
    float shadowContrast = 1.0; // kappa
    int sampleTurns = 16;       // sampling turns
    bool bilateralBlur = true;

    const int totalLights = LIGHT_GRID_WIDTH * LIGHT_GRID_WIDTH * LIGHT_GRID_HEIGHT;
    // initialize point lights
    configurePointLights(modelMatrices, modelColorSizes, pointLightRadius, pointLightSeparation, pointLightVerticalOffset);
    
    // configure instanced array of model transform matrices
    // -------------------------
    glGenBuffers(1, &matrixBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, matrixBuffer);
    glBufferData(GL_ARRAY_BUFFER, totalLights * sizeof(glm::mat4), &modelMatrices[0], GL_STATIC_DRAW);

    // light model has only one mesh
    unsigned int VAO = lightModel.meshes[0].VAO;
    glBindVertexArray(VAO);

    // set attribute pointers for matrix (4 times vec4)
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)0);
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(sizeof(glm::vec4)));
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(2 * sizeof(glm::vec4)));
    glEnableVertexAttribArray(6);
    glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(glm::mat4), (void*)(3 * sizeof(glm::vec4)));

    glVertexAttribDivisor(3, 1);
    glVertexAttribDivisor(4, 1);
    glVertexAttribDivisor(5, 1);
    glVertexAttribDivisor(6, 1);

    // configure instanced array of light colors
    // -------------------------
    unsigned int colorSizeBuffer;
    glGenBuffers(1, &colorSizeBuffer);
   
    glBindVertexArray(VAO);
    // set attribute pointers for light color + radius (vec4)
    glEnableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, colorSizeBuffer);
    glBufferData(GL_ARRAY_BUFFER, totalLights * sizeof(glm::vec4), &modelColorSizes[0], GL_STATIC_DRAW);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), (void*)0);
    glVertexAttribDivisor(2, 1);

    glBindVertexArray(0);
    
    // shader configuration
    // --------------------
    pbrShader.use();
    pbrShader.setUniformInt("gPosition", 0);
    pbrShader.setUniformInt("gNormal", 1);
    pbrShader.setUniformInt("gDiffuse", 2);
    pbrShader.setUniformInt("gSpecular", 3);
    pbrShader.setUniformInt("shadowSAT", 4);
    pbrShader.setUniformInt("environmentMap", 5);
    pbrShader.setUniformInt("irradianceMap", 6);
    pbrShader.setUniformInt("brdfLUT", 7);
    pbrShader.setUniformInt("ambientOcclusion", 8);
    pbrShader.setUniformInt("shadowMap", 9);
    pbrShader.setUniformInt("iblSamples", iblSamples);

    // deferred point lighting shader
    shaderPointLightingPass.use();
    shaderPointLightingPass.setUniformInt("gPosition", 0);
    shaderPointLightingPass.setUniformInt("gNormal", 1);
    shaderPointLightingPass.setUniformInt("gDiffuse", 2);
    shaderPointLightingPass.setUniformInt("gSpecular", 3);
    shaderPointLightingPass.setUniformVec2f("screenSize", SCR_WIDTH, SCR_HEIGHT);

    // G-Buffer debug shader
    shaderGBufferDebug.use();
    shaderGBufferDebug.setUniformInt("gPosition", 0);
    shaderGBufferDebug.setUniformInt("gNormal", 1);
    shaderGBufferDebug.setUniformInt("gDiffuse", 2);
    shaderGBufferDebug.setUniformInt("gSpecular", 3);
    shaderGBufferDebug.setUniformInt("gBufferMode", 1);

    // SSAO debug shader
    shaderSSAODebug.use();
    shaderSSAODebug.setUniformInt("aoTexture", 0);

    // SSAO generation shader
    shaderSSAO.use();
    shaderSSAO.setUniformInt("gPosition", 0);
    shaderSSAO.setUniformInt("gNormal", 1);

    // SAT generation shader
    shaderSATHorizontal.use();
    shaderSATHorizontal.setUniformInt("image", 0);
    shaderSATVertical.use();
    shaderSATVertical.setUniformInt("image", 0);

    computeSAT.use();
    computeSAT.setUniformInt("input_image", 0);
    computeSAT.setUniformInt("output_image", 1);


    OpenGLBlurData data(8, 8.0f);
    // Create our blur data uniform buffer object
    unsigned int uboBlurData;
    glGenBuffers(1, &uboBlurData);
    glBindBuffer(GL_UNIFORM_BUFFER, uboBlurData);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(OpenGLBlurData), &data, GL_STATIC_DRAW);
    glBindBuffer(GL_UNIFORM_BUFFER, 0);

    // SSAO compute bilateral blur
    computeBilateralBlur.use();
    unsigned int block_index = glGetUniformBlockIndex(computeBilateralBlur.ID, "Blur");
    glUniformBlockBinding(computeBilateralBlur.ID, block_index, 7);
    computeBilateralBlur.setUniformInt("uSrc", 0);
    computeBilateralBlur.setUniformInt("uDst", 1);
    computeBilateralBlur.setUniformVec2f("screenSize", SCR_WIDTH, SCR_HEIGHT);
    computeBilateralBlur.setUniformInt("gPosition", 2);
    computeBilateralBlur.setUniformInt("gNormal", 3);

    // cubemap render shader
    cubemapShader.use();
    cubemapShader.setUniformInt("environmentMap", 0);

    // Shadow texture debug shader
    shaderDebugDepthMap.use();
    shaderDebugDepthMap.setUniformInt("depthMap", 0);

    // Cubemap texture debug shader
    shaderDebugCubemap.use();
    shaderDebugCubemap.setUniformInt("cubeMap", 0);

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // per-frame time logic
        // --------------------
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glEnable(GL_DEPTH_TEST);

        // 1. render depth of scene to texture (from light's perspective)
        // --------------------------------------------------------------
        glm::mat4 lightProjection, lightView;
        glm::mat4 lightSpaceMatrix;
        glm::mat4 model = glm::mat4(1.0f);
        float zNear = 1.0f, zFar = 15.0f;

        if (enableShadows) {
            lightProjection = glm::ortho(-10.0f, 10.0f, -10.0f, 10.0f, zNear, zFar);
            glm::vec3 lightPosition = arcballLight.eye();
            lightView = glm::lookAt(lightPosition, glm::vec3(0.0f), glm::vec3(0.0, 1.0, 0.0));
            lightSpaceMatrix = lightProjection * lightView;
            // render scene from light's point of view
            shaderDepthWrite.use();
            shaderDepthWrite.setUniformMat4("lightSpaceMatrix", lightSpaceMatrix);
            shaderDepthWrite.setUniformMat4("model", model);

            glViewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
            sBuffer.bindOutput();
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            // render the textured floor
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, woodTexture);
            glBindVertexArray(planeVAO);
            glDrawArrays(GL_TRIANGLES, 0, 6);

            for (unsigned int i = 0; i < objectPositions.size(); i++)
            {
                model = glm::mat4(1.0f);
                model = glm::translate(model, objectPositions[i]);
                model = glm::scale(model, glm::vec3(modelScale));
                shaderDepthWrite.setUniformMat4("model", model);
                meshModels[i]->draw(shaderDepthWrite);
            }
            FrameBuffer::unbind();

            // perform shadow map blurring 
            int width = (int)SHADOW_MAP_SIZE;
            int height = (int)SHADOW_MAP_SIZE;

            // compute shader SAT generation as described in OpenGL SuperBible 7th Edition (CH 10)
            computeSAT.use();
            // bind shadow buffer as first texture
            sBuffer.bindImage(0, 0, GL_RGBA32F);
            satBuffer.bindImage(1, 0, GL_RGBA32F);
            glDispatchCompute(width, 1, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
            satBuffer.bindImage(0, 0, GL_RGBA32F);
            satBuffer.bindImage(1, 1, GL_RGBA32F);
            glDispatchCompute(width, 1, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);


            // SAT Generation as developed by Hensley
            /*int maxIterations = glm::log2(float(SHADOW_MAP_SIZE));
            for (int iteration = 0; iteration < maxIterations; ++iteration)
            {
                glViewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
                satBufferA.bindOutput();
                glClearColor(1, 1, 1, 1);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                shaderSATHorizontal.use();
                shaderSATHorizontal.setUniformInt("iteration", iteration);
                if (iteration == 0) {
                    glActiveTexture(GL_TEXTURE0);
                    sBuffer.bindInput(0);
                }
                else {
                    satBufferB.bindInput();
                }
                renderQuad();
                FrameBuffer::unbind();

                glViewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
                satBufferB.bindOutput();
                glClearColor(1, 1, 1, 1);
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
                shaderSATVertical.use();
                shaderSATVertical.setUniformInt("iteration", iteration);
                satBufferA.bindInput();
                renderQuad();
                FrameBuffer::unbind();
            }
            */
        }
        else {
            // just clear the depth texture if shadows aren't being generated
            glViewport(0, 0, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
            sBuffer.bindOutput();
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        }
        
        // 2. geometry pass: render scene's geometry/color data into gbuffer
        // -----------------------------------------------------------------
        // reset viewport
        glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
        gBuffer.bindOutput();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 150.0f);
        glm::mat4 view = arcballCamera.transform();
        model = glm::mat4(1.0f);
        cubemapShader.use();
        cubemapShader.setUniformMat4("projection", projection);

        shaderTexturedGeometryPass.use();
        shaderTexturedGeometryPass.setUniformMat4("projection", projection);
        shaderTexturedGeometryPass.setUniformMat4("view", view);
        shaderTexturedGeometryPass.setUniformMat4("model", model);
        glm::vec4 floorSpecular = glm::vec4(0.5f, 0.5f, 0.5f, 0.0f);
        shaderTexturedGeometryPass.setUniformVec4f("specularCol", floorSpecular);
        // render the textured floor
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, woodTexture);
        glBindVertexArray(planeVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // render non-textured models
        shaderGeometryPass.use();
        shaderGeometryPass.setUniformMat4("projection", projection);
        shaderGeometryPass.setUniformMat4("view", view);
        glm::vec4 specular = glm::vec4(1.0f, 1.0f, 1.0f, 0.1f);
       
        for (unsigned int i = 0; i < objectPositions.size(); i++)
        {
            model = glm::mat4(1.0f);
            model = glm::translate(model, objectPositions[i]);
            model = glm::scale(model, glm::vec3(modelScale));
            shaderGeometryPass.setUniformMat4("model", model);
            glm::vec4 diffuse = glm::vec4(materials[i].diffuse, materials[i].roughness);
            glm::vec4 specular = glm::vec4(materials[i].specular, materials[i].metallic);
            shaderGeometryPass.setUniformVec4f("diffuseCol", diffuse);
            shaderGeometryPass.setUniformVec4f("specularCol", specular);
            meshModels[i]->draw(shaderGeometryPass);
        }
        FrameBuffer::unbind();

        // 2a. generate SSAO texture
        // ------------------------
        aoBuffer.bindOutput();
        glClear(GL_COLOR_BUFFER_BIT);
        shaderSSAO.use();
        shaderSSAO.setUniformMat4("view", view);
        shaderSSAO.setUniformInt("aoSamples", aoSamples);
        shaderSSAO.setUniformFloat("sampleRadius", sampleRadius);
        shaderSSAO.setUniformInt("sampleTurns", sampleTurns);
        shaderSSAO.setUniformFloat("shadowScalar", shadowScalar);
        shaderSSAO.setUniformFloat("shadowContrast", shadowContrast);
        gBuffer.bindInput();
        renderQuad();
        FrameBuffer::unbind();

        // blur AO texture
        if (bilateralBlur)
        {
            computeBilateralBlur.use();
            glBindBufferBase(GL_UNIFORM_BUFFER, 7, uboBlurData);
            computeBilateralBlur.setUniformMat4("projection", projection);
            computeBilateralBlur.setUniformMat4("view", view);
            aoBuffer.bindImage(0, 0, GL_RGBA32F, GL_READ_ONLY);
            aoBuffer.bindImage(1, 1, GL_RGBA32F, GL_WRITE_ONLY);
            computeBilateralBlur.setUniformVec2i("direction", 1, 0);
            glActiveTexture(GL_TEXTURE2);
            gBuffer.bindInput(0);
            glActiveTexture(GL_TEXTURE3);
            gBuffer.bindInput(1);
            glDispatchCompute(std::ceil(float(SCR_WIDTH) / 128), SCR_HEIGHT, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

            aoBuffer.bindImage(0, 1, GL_RGBA32F, GL_READ_ONLY);
            aoBuffer.bindImage(1, 0, GL_RGBA32F, GL_WRITE_ONLY);
            computeBilateralBlur.setUniformVec2i("direction", 0, 1);
            glDispatchCompute(std::ceil(float(SCR_HEIGHT) / 128), SCR_WIDTH, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
        }

        // 3. lighting pass: calculate lighting by iterating over a screen filled quad pixel-by-pixel using the gbuffer's content and shadow map
        // -----------------------------------------------------------------------------------------------------------------------
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (gBufferMode == GBufferRender::Final)
        {
            pbrShader.use();
            // bind all of our input textures
            gBuffer.bindInput();

            // bind depth texture
            glActiveTexture(GL_TEXTURE4);
            //sBuffer.bindTex(1);
            satBuffer.bindInput(1);
            glActiveTexture(GL_TEXTURE5);
            glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
            glActiveTexture(GL_TEXTURE6);
            glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap);
            glActiveTexture(GL_TEXTURE7);
            glBindTexture(GL_TEXTURE_2D, brdfLUTTexture);
            glActiveTexture(GL_TEXTURE8);
            aoBuffer.bindInput(0);
            glActiveTexture(GL_TEXTURE9);
            sBuffer.bindInput(0);

            glm::vec3 lightPosition = arcballLight.eye();
            pbrShader.setUniformVec3f("gLight.Position", lightPosition);
            pbrShader.setUniformVec3f("gLight.Color", globalLight.color);
            pbrShader.setUniformFloat("gLight.Intensity", globalLight.intensity);

            glm::vec3 camPosition = arcballCamera.eye();
            pbrShader.setUniformVec3f("viewPos", camPosition);
            pbrShader.setUniformMat4("lightSpaceMatrix", lightSpaceMatrix);
            pbrShader.setUniformInt("iblSamples", iblSamples);
            pbrShader.setUniformFloat("shadowSaturation", shadowSaturation);
            pbrShader.setUniformFloat("PenumbraSize", penumbraSize);
            pbrShader.setUniformInt("lightSourceRadius", lightSourceRadius);
            pbrShader.setUniformFloat("zNear", zNear);
            pbrShader.setUniformFloat("zFar", zFar);
            pbrShader.setUniformBool("softSATVSM", softSATVSM);
        }
        else if (gBufferMode == GBufferRender::Occlusion)
        {
            shaderSSAODebug.use();
            glActiveTexture(GL_TEXTURE0);
            aoBuffer.bindInput(0);
        }
        else // for G-Buffer debuging 
        {
            shaderGBufferDebug.use();
            shaderGBufferDebug.setUniformInt("gBufferMode", gBufferMode);
            // bind all of our input textures
            gBuffer.bindInput();
        }
        
        // finally render quad
        renderQuad();

        static bool colorSizeBufferDirty = false;

        // 3.5 lighting pass: render point lights on top of main scene with additive blending and utilizing G-Buffer for lighting.
        // -----------------------------------------------------------------------------------------------------------------------
        if (gBufferMode == GBufferRender::Final) {
            // TODO: Disable the point lights for now
            /*
            shaderPointLightingPass.use();
            gBuffer.bindInput();
            shaderPointLightingPass.setUniformMat4("projection", projection);
            shaderPointLightingPass.setUniformMat4("view", view);

            glEnable(GL_CULL_FACE);
            // only render the back faces of the light volume spheres
            glFrontFace(GL_CW);
            glDisable(GL_DEPTH_TEST);
            // enable additive blending
            glEnable(GL_BLEND);
            glBlendFunc(GL_ONE, GL_ONE);
            glm::vec3 camPosition = arcballCamera.eye();
            shaderPointLightingPass.setUniformVec3f("viewPos", camPosition);
            shaderPointLightingPass.setUniformFloat("lightIntensity", pointLightIntensity);
            shaderPointLightingPass.setUniformFloat("glossiness", glossiness);
            glBindVertexArray(lightModel.meshes[0].VAO);
            // don't update the color and size buffer every frame
            if (colorSizeBufferDirty) {
                glBindBuffer(GL_ARRAY_BUFFER, colorSizeBuffer);
                glBufferData(GL_ARRAY_BUFFER, LIGHT_GRID_WIDTH * LIGHT_GRID_WIDTH * LIGHT_GRID_HEIGHT * sizeof(glm::vec4), &modelColorSizes[0], GL_STATIC_DRAW);
            }
            glDrawElementsInstanced(GL_TRIANGLES, lightModel.meshes[0].indices.size(), GL_UNSIGNED_INT, 0, totalLights);
            glBindVertexArray(0);

            glDisable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glFrontFace(GL_CCW);
            glDisable(GL_CULL_FACE);
            */
        }

        // render cubemap with depth testing enabled
        if (gBufferMode == GBufferRender::Final) { 
            // copy content of geometry's depth buffer to default framebuffer's depth buffer
            // ----------------------------------------------------------------------------------
            gBuffer.bindRead();
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); // write to default framebuffer
            // blit to default framebuffer. 
            glBlitFramebuffer(0, 0, SCR_WIDTH, SCR_HEIGHT, 0, 0, SCR_WIDTH, SCR_HEIGHT, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
            // unbind framebuffer for now
            FrameBuffer::unbind();

            glEnable(GL_DEPTH_TEST);
            cubemapShader.use();
            cubemapShader.setUniformMat4("view", view);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
            renderCube();
        }

        // strictly used for debugging point light volumes (sizes, positions, etc)
        if (drawPointLights && gBufferMode == GBufferRender::Final) {
            // re-enable the depth testing 
            glEnable(GL_DEPTH_TEST);

            // render lights on top of scene with Z-testing
            // --------------------------------
            shaderLightSphere.use();
            shaderLightSphere.setUniformMat4("projection", projection);
            shaderLightSphere.setUniformMat4("view", view);

            glPolygonMode(GL_FRONT_AND_BACK, drawPointLightsWireframe ? GL_LINE : GL_FILL);
            glBindVertexArray(lightModel.meshes[0].VAO);
            glDrawElementsInstanced(GL_TRIANGLES, lightModel.meshes[0].indices.size(), GL_UNSIGNED_INT, 0, totalLights);
            glBindVertexArray(0);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

            shaderGlobalLightSphere.use();
            shaderGlobalLightSphere.setUniformMat4("projection", projection);
            shaderGlobalLightSphere.setUniformMat4("view", view);
            // render the global light model
            model = glm::mat4(1.0f);
            model = glm::translate(model, arcballLight.eye());
            shaderGlobalLightSphere.setUniformMat4("model", model);
            shaderGlobalLightSphere.setUniformVec3f("lightColor", globalLight.color);
            shaderGlobalLightSphere.setUniformFloat("lightRadius", globalLight.radius);
            lightModel.draw(shaderGlobalLightSphere);
        }

        if (showDepthMap) {
            // render Depth map to quad for visual debugging
            // ---------------------------------------------
            model = glm::mat4(1.0f);
            //model = glm::translate(model, glm::vec3(0.7f, -0.7f, 0.0f));
            //model = glm::scale(model, glm::vec3(0.3f, 0.3f, 1.0f)); // Make it 30% of total screen size
            shaderDebugDepthMap.use();
            shaderDebugDepthMap.setUniformMat4("transform", model);
            shaderDebugDepthMap.setUniformFloat("zNear", zNear);
            shaderDebugDepthMap.setUniformFloat("zFar", zFar);
            glActiveTexture(GL_TEXTURE0);
            //sBuffer.bindInput(1);
            satBuffer.bindInput(1);
            renderQuad();

            /*shaderDebugCubemap.use();
            shaderDebugCubemap.setUniformMat4("transform", model);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap);
            renderQuad();*/
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("Controls");                          // Create a window called "Controls" and append into it.

            if (ImGui::CollapsingHeader("SSAO")) {
                ImGui::SliderInt("Random samples", &aoSamples, 1, 64);
                ImGui::SliderFloat("Sample radius", &sampleRadius, 0.0f, 5.0f);
                ImGui::SliderInt("Sample turns", &sampleTurns, 1, 64);
                ImGui::SliderFloat("Intensity scale", &shadowScalar, 0.1f, 20.0f);
                ImGui::SliderFloat("Contrast", &shadowContrast, 0.1f, 10.0f);
                ImGui::Checkbox("Bilateral Blur", &bilateralBlur);
            }

            if (ImGui::CollapsingHeader("IBL")) {
                ImGui::SliderInt("Random samples", &iblSamples, 1, 100);
                // list of cubemaps
                const char* cubeMaps[] = { "Newport Loft", "Tropical Beach", "Alexs Apartment", "Malibu Overloop", "Tokyo BigSight", "Barcelona Rooftops", "Winter Forest", "Ueno Shrine" };
                if (ImGui::Combo("Skybox", &CubemapSelection, cubeMaps, IM_ARRAYSIZE(cubeMaps))) {
                    renderCubemap(CubemapSelection, equirectangularToCubemapShader, irradianceShader);
                }
            }

            if (ImGui::CollapsingHeader("Materials Config")) {
                if (ImGui::CollapsingHeader("Model 1")) {
                    ImGui::PushID(1);
                    ImGui::ColorEdit3("Diffuse (Kd)", (float*)&materials[0].diffuse);   // Edit 3 floats representing Kd color (r, g, b)
                    ImGui::ColorEdit3("Specular (Ks)", (float*)&materials[0].specular); // Edit 3 floats representing Ks color (r, g, b)
                    ImGui::SliderFloat("Roughness", &materials[0].roughness, 0.0, 1.0f);
                    ImGui::SliderFloat("Metallic", &materials[0].metallic, 0.0, 1.0f);
                    ImGui::PopID();
                } 
                if (ImGui::CollapsingHeader("Model 2")) {
                    ImGui::PushID(2);
                    ImGui::ColorEdit3("Diffuse (Kd)", (float*)&materials[1].diffuse);   // Edit 3 floats representing Kd color (r, g, b)
                    ImGui::ColorEdit3("Specular (Ks)", (float*)&materials[1].specular); // Edit 3 floats representing Ks color (r, g, b)
                    ImGui::SliderFloat("Roughness", &materials[1].roughness, 0.0, 1.0f);
                    ImGui::SliderFloat("Metallic", &materials[1].metallic, 0.0, 1.0f);
                    ImGui::PopID();
                }
                if (ImGui::CollapsingHeader("Model 3")) {
                    ImGui::PushID(3);
                    ImGui::ColorEdit3("Diffuse (Kd)", (float*)&materials[2].diffuse);   // Edit 3 floats representing Kd color (r, g, b)
                    ImGui::ColorEdit3("Specular (Ks)", (float*)&materials[2].specular); // Edit 3 floats representing Ks color (r, g, b)
                    ImGui::SliderFloat("Roughness", &materials[2].roughness, 0.0, 1.0f);
                    ImGui::SliderFloat("Metallic", &materials[2].metallic, 0.0, 1.0f);
                    ImGui::PopID();
                }
            }
            if (ImGui::CollapsingHeader("Lighting Config")) {
                if (ImGui::CollapsingHeader("Global Light")) {
                    ImGui::ColorEdit3("Color", (float*)&globalLight.color, ImGuiColorEditFlags_HDR);   // Edit 3 floats representing Kd color (r, g, b) 
                    ImGui::SliderFloat("Intensity", (float*)&globalLight.intensity, 0.0f, 15.0f, "%.3f");
                }

                if (ImGui::CollapsingHeader("Point Lights")) {
                    ImGui::SliderFloat("Intensity", &pointLightIntensity, 0.0f, 10.0f, "%.3f");
                    if (ImGui::SliderFloat("Radius", &pointLightRadius, 0.3f, 2.5f, "%.3f")) {
                        updatePointLights(modelMatrices, modelColorSizes, pointLightSeparation, pointLightVerticalOffset, pointLightRadius);
                        colorSizeBufferDirty = true;
                    }
                    else {
                        colorSizeBufferDirty = false;
                    }
                    if (ImGui::SliderFloat("Separation", &pointLightSeparation, 0.4f, 1.5f, "%.3f")) {
                        updatePointLights(modelMatrices, modelColorSizes, pointLightSeparation, pointLightVerticalOffset, pointLightRadius);
                    }
                    if (ImGui::SliderFloat("Vertical Offset", &pointLightVerticalOffset, -2.0f, 3.0f)) {
                        updatePointLights(modelMatrices, modelColorSizes, pointLightSeparation, pointLightVerticalOffset, pointLightRadius);
                    }
                }

                // Shadows
                if (ImGui::CollapsingHeader("Shadows")) {
                    ImGui::Checkbox("Enabled", &enableShadows);
                    ImGui::SliderFloat("Saturation", &shadowSaturation, 0.0f, 1.0f, "%.2f");
                    ImGui::SliderFloat("Penumbra", &penumbraSize, 0.5f, 10.0f, "%.4f");
                    ImGui::SliderInt("Light radius", &lightSourceRadius, 4, 40);
                    ImGui::Checkbox("Contact-hardening", &softSATVSM);
                }
            }
            if (ImGui::CollapsingHeader("Debug")) {
                const char* gBuffers[] = { "Final render", "Position (world)", "Normal (world)", "Diffuse", "Specular", "Occlusion"};
                ImGui::Combo("G-Buffer View", &gBufferMode, gBuffers, IM_ARRAYSIZE(gBuffers));
                pbrShader.setUniformInt("gBufferMode", gBufferMode);
                ImGui::Checkbox("Point lights volumes", &drawPointLights);
                ImGui::SameLine(); ImGui::Checkbox("Wireframe", &drawPointLightsWireframe);
                ImGui::Checkbox("Show depth texture", &showDepthMap);
                ImGui::Text("Mouse Controls:");
                ImGui::RadioButton("Camera", &mouseControl, 0); ImGui::SameLine();
                ImGui::RadioButton("Light", &mouseControl, 1);
            }
                                                                    
            //ImGui::ShowDemoWindow();

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::Text("Point lights in scene: %i", LIGHT_GRID_WIDTH * LIGHT_GRID_WIDTH * LIGHT_GRID_HEIGHT);
            ImGui::End();

        }

        // Rendering
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &planeVAO);
    glDeleteBuffers(1, &planeVBO);

    glfwTerminate();
    return 0;

}

// Node: separation < 1.0 will cause lights to penetrate each other, and > 1.0 they will separate (1.0 is just touching)
void configurePointLights(std::vector<glm::mat4>& modelMatrices, std::vector<glm::vec4>& modelColorSizes, float radius, float separation, float yOffset)
{
    srand(glfwGetTime());
    // add some uniformly spaced point lights
    for (unsigned int lightIndexX = 0; lightIndexX < LIGHT_GRID_WIDTH; lightIndexX++)
    {
        for (unsigned int lightIndexZ = 0; lightIndexZ < LIGHT_GRID_WIDTH; lightIndexZ++)
        {
            for (unsigned int lightIndexY = 0; lightIndexY < LIGHT_GRID_HEIGHT; lightIndexY++)
            {
                float diameter = 2.0f * radius;
                float xPos = (lightIndexX - (LIGHT_GRID_WIDTH - 1.0f) / 2.0f) * (diameter * separation);
                float zPos = (lightIndexZ - (LIGHT_GRID_WIDTH - 1.0f) / 2.0f) * (diameter * separation);
                float yPos = (lightIndexY - (LIGHT_GRID_HEIGHT - 1.0f) / 2.0f) * (diameter * separation) + yOffset;
                double angle = double(rand()) * 2.0 * glm::pi<float>() / (double(RAND_MAX));
                double length = double(rand()) * 0.5 / (double(RAND_MAX));
                float xOffset = cos(angle) * length;
                float zOffset = sin(angle) * length;
                xPos += xOffset;
                zPos += zOffset;
                // also calculate random color
                float rColor = ((rand() % 100) / 200.0f) + 0.5; // between 0.5 and 1.0
                float gColor = ((rand() % 100) / 200.0f) + 0.5; // between 0.5 and 1.0
                float bColor = ((rand() % 100) / 200.0f) + 0.5; // between 0.5 and 1.0

                int curLight = lightIndexX * LIGHT_GRID_WIDTH * LIGHT_GRID_HEIGHT + lightIndexZ * LIGHT_GRID_HEIGHT + lightIndexY;
                glm::mat4 model = glm::mat4(1.0f);
                model = glm::translate(model, glm::vec3(xPos, yPos, zPos));
                // now add to list of matrices
                modelMatrices.emplace_back(model);
                modelColorSizes.emplace_back(glm::vec4(rColor, gColor, bColor, radius));
            }
        }
    }
}

void updatePointLights(std::vector<glm::mat4>& modelMatrices, std::vector<glm::vec4>& modelColorSizes, float separation, float yOffset, float radius)
{
    if (separation < 0.0f) {
        return;
    }
    // add some uniformly spaced point lights
    for (unsigned int lightIndexX = 0; lightIndexX < LIGHT_GRID_WIDTH; lightIndexX++)
    {
        for (unsigned int lightIndexZ = 0; lightIndexZ < LIGHT_GRID_WIDTH; lightIndexZ++)
        {
            for (unsigned int lightIndexY = 0; lightIndexY < LIGHT_GRID_HEIGHT; lightIndexY++)
            {
                int curLight = lightIndexX * LIGHT_GRID_WIDTH * LIGHT_GRID_HEIGHT + lightIndexZ * LIGHT_GRID_HEIGHT + lightIndexY;
                float diameter = 2.0f * INITIAL_POINT_LIGHT_RADIUS;
                float xPos = (lightIndexX - (LIGHT_GRID_WIDTH - 1.0f) / 2.0f) * (diameter * separation);
                float zPos = (lightIndexZ - (LIGHT_GRID_WIDTH - 1.0f) / 2.0f) * (diameter * separation);
                float yPos = (lightIndexY - (LIGHT_GRID_HEIGHT - 1.0f) / 2.0f) * (diameter * separation);
                
                // modify matrix translation
                modelMatrices[curLight][3] = glm::vec4(xPos, yPos + yOffset, zPos, 1.0);
                modelColorSizes[curLight].w = radius;
            }
        }
    }

    // update the instance matrix buffer
    glBindBuffer(GL_ARRAY_BUFFER, matrixBuffer);
    glBufferData(GL_ARRAY_BUFFER, LIGHT_GRID_WIDTH * LIGHT_GRID_WIDTH * LIGHT_GRID_HEIGHT * sizeof(glm::mat4), &modelMatrices[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}


// renderQuad() renders a 1x1 XY quad in NDC
// -----------------------------------------
unsigned int quadVAO = 0;
unsigned int quadVBO;
void renderQuad()
{
    if (quadVAO == 0)
    {
        float quadVertices[] = {
            // positions        // texture Coords
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
             1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
             1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        };
        // setup plane VAO
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}

// renderCube() renders a 1x1 3D cube in NDC.
// -------------------------------------------------
unsigned int cubeVAO = 0;
unsigned int cubeVBO = 0;
void renderCube()
{
    // initialize (if necessary)
    if (cubeVAO == 0)
    {
        float vertices[] = {
            // back face
            -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
             1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
             1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f, // bottom-right         
             1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
            -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
            -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f, // top-left
            // front face
            -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
             1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f, // bottom-right
             1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
             1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
            -1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f, // top-left
            -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
            // left face
            -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
            -1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-left
            -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
            -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
            -1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-right
            -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
            // right face
             1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
             1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
             1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-right         
             1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
             1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
             1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-left     
            // bottom face
            -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
             1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f, // top-left
             1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
             1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
            -1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, // bottom-right
            -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
            // top face
            -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
             1.0f,  1.0f , 1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
             1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f, // top-right     
             1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
            -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
            -1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f  // bottom-left        
        };
        glGenVertexArrays(1, &cubeVAO);
        glGenBuffers(1, &cubeVBO);
        // fill buffer
        glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
        // link vertex attributes
        glBindVertexArray(cubeVAO);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
    // render Cube
    glBindVertexArray(cubeVAO);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    ImGuiIO& io = ImGui::GetIO();

    // only rotate the camera if we aren't over imGui
    if (leftMouseButtonPressed && !io.WantCaptureMouse) {
        //std::cout << "Xpos = " << xpos << ", Ypos = " << ypos << std::endl;
        float prevMouseX = 2.0f * lastX / SCR_WIDTH - 1;
        float prevMouseY = -1.0f * (2.0f * lastY / SCR_HEIGHT - 1);
        float curMouseX = 2.0f * xpos / SCR_WIDTH - 1;
        float curMouseY = -1.0f * (2.0f * ypos / SCR_HEIGHT - 1);
        if (mouseControl == 1) { // apply rotation to the global light
            arcballLight.rotate(glm::vec2(prevMouseX, prevMouseY), glm::vec2(curMouseX, curMouseY));
        }
        else {
            arcballCamera.rotate(glm::vec2(prevMouseX, prevMouseY), glm::vec2(curMouseX, curMouseY));
        } 
    }

    // pan the camera when the right mouse is pressed
    if (rightMouseButtonPressed && !io.WantCaptureMouse) {
        float prevMouseX = 2.0f * lastX / SCR_WIDTH - 1;
        float prevMouseY = -1.0f * (2.0f * lastY / SCR_HEIGHT - 1);
        float curMouseX = 2.0f * xpos / SCR_WIDTH - 1;
        float curMouseY = -1.0f * (2.0f * ypos / SCR_HEIGHT - 1);
        glm::vec2 mouseDelta = glm::vec2(curMouseX - prevMouseX, curMouseY - prevMouseY);
        arcballCamera.pan(mouseDelta);
    }

    lastX = xpos;
    lastY = ypos;
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        leftMouseButtonPressed = true;
    }
    else if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        leftMouseButtonPressed = false;
    }

    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        rightMouseButtonPressed = true;
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) {
        rightMouseButtonPressed = false;
    }
      
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantCaptureMouse) {
        return;
    }

    float distanceSq = glm::distance2(arcballCamera.center(), arcballCamera.eye());
    if (distanceSq < MAX_CAMERA_DISTANCE && yoffset < 0)
    {
        // zoom out
        arcballCamera.zoom(yoffset);
    }
    else if (yoffset > 0)
    {
        // zoom in
        arcballCamera.zoom(yoffset);
    }
}

// utility function for loading a 2D texture from file
// ---------------------------------------------------
unsigned int loadTexture(char const * path, bool gammaCorrection)
{
    unsigned int textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    unsigned char *data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data)
    {
        GLenum internalFormat;
        GLenum dataFormat;
        if (nrComponents == 1)
        {
            internalFormat = dataFormat = GL_RED;
        }
        else if (nrComponents == 3)
        {
            internalFormat = gammaCorrection ? GL_SRGB : GL_RGB;
            dataFormat = GL_RGB;
        }
        else if (nrComponents == 4)
        {
            internalFormat = gammaCorrection ? GL_SRGB_ALPHA : GL_RGBA;
            dataFormat = GL_RGBA;
        }

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, dataFormat, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, internalFormat == GL_RGBA ? GL_CLAMP_TO_EDGE : GL_REPEAT); // for this tutorial: use GL_CLAMP_TO_EDGE to prevent semi-transparent borders. Due to interpolation it takes texels from next repeat 
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, internalFormat == GL_RGBA ? GL_CLAMP_TO_EDGE : GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else
    {
        std::cout << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}

void renderCubemap(int cubemap, Shader& equirectangularToCubemapShader, Shader& irradianceShader) {

    static const std::string hdrCubemaps[] = {
        PATH + "/OpenGL/images/newport_loft.hdr",
        PATH + "/OpenGL/images/tropical_beach.hdr",
        PATH + "/OpenGL/images/alexs_apartment.hdr",
        PATH + "/OpenGL/images/malibu_overlook.hdr",
        PATH + "/OpenGL/images/tokyo_bigsight.hdr",
        PATH + "/OpenGL/images/barcelona_rooftops.hdr",
        PATH + "/OpenGL/images/winter_forest.hdr",
        PATH + "/OpenGL/images/ueno_shrine.hdr"
    };

    int width, height, nrComponents;
    float *data = stbi_loadf(hdrCubemaps[cubemap].c_str(), &width, &height, &nrComponents, 0);
    if (data)
    {
        if (hdrTexture == 0) {
            glGenTextures(1, &hdrTexture);
        }
        glBindTexture(GL_TEXTURE_2D, hdrTexture);
        // load a floating point HDR texture data
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, data);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else
    {
        std::cout << "Failed to load HDR image." << std::endl;
    }

    // pbr: setup cubemap to render to and attach to framebuffer
    // ---------------------------------------------------------
    if (envCubemap == 0) {
        glGenTextures(1, &envCubemap);
    }
    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, 9);
    for (unsigned int i = 0; i < 6; ++i)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, ENV_CUBEMAP_SIZE, ENV_CUBEMAP_SIZE, 0, GL_RGB, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // generate mipmaps for the cubemap so OpenGL automatically allocates the required memory.
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);


    // pbr: set up projection and view matrices for capturing data onto the 6 cubemap face directions
    // ----------------------------------------------------------------------------------------------
    glm::mat4 captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
    glm::mat4 captureViews[] =
    {
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f,  0.0f,  0.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  1.0f,  0.0f), glm::vec3(0.0f,  0.0f,  1.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f,  0.0f), glm::vec3(0.0f,  0.0f, -1.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  0.0f,  1.0f), glm::vec3(0.0f, -1.0f,  0.0f)),
        glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f,  0.0f, -1.0f), glm::vec3(0.0f, -1.0f,  0.0f))
    };

    // pbr: convert HDR equirectangular environment map to cubemap equivalent and generate mip levels
    // ----------------------------------------------------------------------
    equirectangularToCubemapShader.use();
    equirectangularToCubemapShader.setUniformInt("equirectangularMap", 0);
    equirectangularToCubemapShader.setUniformMat4("projection", captureProjection);

    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    unsigned int maxMipLevels = 9;
    for (unsigned int mip = 0; mip < maxMipLevels; ++mip)
    {
        // resize framebuffer according to mip-level size.
        unsigned int mipWidth = ENV_CUBEMAP_SIZE * std::pow(0.5, mip);
        unsigned int mipHeight = ENV_CUBEMAP_SIZE * std::pow(0.5, mip);
        glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, mipWidth, mipHeight);
        glViewport(0, 0, mipWidth, mipHeight);
        for (unsigned int i = 0; i < 6; ++i)
        {
            equirectangularToCubemapShader.setUniformMat4("view", captureViews[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, envCubemap, mip);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            renderCube();
        }
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // pbr: create an irradiance cubemap, and re-scale capture FBO to irradiance scale.
    // --------------------------------------------------------------------------------
    if (irradianceMap == 0) {
        glGenTextures(1, &irradianceMap);
    }
    glBindTexture(GL_TEXTURE_CUBE_MAP, irradianceMap);
    for (unsigned int i = 0; i < 6; ++i)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, IRRADIANCE_CUBEMAP_SIZE, IRRADIANCE_CUBEMAP_SIZE, 0, GL_RGB, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, captureRBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, IRRADIANCE_CUBEMAP_SIZE, IRRADIANCE_CUBEMAP_SIZE);

    // pbr: solve diffuse integral by convolution to create an irradiance (cube)map.
    // -----------------------------------------------------------------------------
    irradianceShader.use();
    irradianceShader.setUniformInt("environmentMap", 0);
    irradianceShader.setUniformMat4("projection", captureProjection);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
    // set the viewport to match the cubemap size
    glViewport(0, 0, IRRADIANCE_CUBEMAP_SIZE, IRRADIANCE_CUBEMAP_SIZE);
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    for (unsigned int i = 0; i < 6; ++i)
    {
        irradianceShader.setUniformMat4("view", captureViews[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, irradianceMap, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        renderCube();
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

}

