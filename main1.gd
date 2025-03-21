extends Control

@export_file("*.glsl") var shader_file: String
@export_range(128, 2048, 1, "exp") var dimension: int = 1024  # Lower default for mobile

@onready var seed_input: SpinBox = $CenterContainer/VBoxContainer/PanelContainer/VBoxContainer/GridContainer/SeedInput
@onready var heightmap_rect: TextureRect = $CenterContainer/VBoxContainer/PanelContainer2/VBoxContainer/GridContainer/RawHeightmap
@onready var island_rect: TextureRect = $CenterContainer/VBoxContainer/PanelContainer2/VBoxContainer/GridContainer/ComputedHeightmap

var noise: FastNoiseLite
var gradient: Gradient
var gradient_tex: GradientTexture1D

var po2_dimensions: int
var start_time: int

var rd: RenderingDevice
var shader_rid: RID
var heightmap_rid: RID
var gradient_rid: RID
var uniform_set: RID
var pipeline: RID

func _init() -> void:
    gradient = Gradient.new()
    gradient.add_point(0.6, Color(0.9, 0.9, 0.9, 1.0))
    gradient.add_point(0.8, Color(1.0, 1.0, 1.0, 1.0))
    gradient.reverse()

    gradient_tex = GradientTexture1D.new()
    gradient_tex.gradient = gradient
    gradient_tex.width = 128  # Smaller gradient texture for mobile

func _ready() -> void:
    randomize_seed()
    po2_dimensions = nearest_po2(dimension)
    noise = FastNoiseLite.new()  # Initialize once here
    noise.noise_type = FastNoiseLite.TYPE_SIMPLEX_SMOOTH
    noise.fractal_octaves = 3  # Reduced from 5 for mobile
    noise.fractal_lacunarity = 1.9
    noise.frequency = 0.005 / (float(po2_dimensions) / 256.0)  # Adjusted for smaller size
    
    $CenterContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/CreateButtonGPU.text += "\n" + RenderingServer.get_video_adapter_name()
    $CenterContainer/VBoxContainer/PanelContainer/VBoxContainer/HBoxContainer/CreateButtonCPU.text += "\n" + OS.get_processor_name()

func _notification(what: int) -> void:
    if what == NOTIFICATION_PREDELETE:
        cleanup_gpu()

func randomize_seed() -> void:
    seed_input.value = randi()

func prepare_image() -> Image:
    start_time = Time.get_ticks_usec()
    noise.seed = int(seed_input.value)
    var heightmap: Image = noise.get_image(po2_dimensions, po2_dimensions, false, false)
    
    # Resize directly for display, avoid cloning
    heightmap.resize(256, 256, Image.INTERPOLATE_NEAREST)  # Smaller preview for mobile
    heightmap_rect.texture = ImageTexture.create_from_image(heightmap)
    
    return heightmap

func init_gpu() -> void:
    rd = RenderingServer.create_local_rendering_device()
    if rd == null:
        OS.alert("RenderingDevice unavailable on this mobile device.")
        return

    shader_rid = load_shader(rd, shader_file)

    var heightmap_format: RDTextureFormat = RDTextureFormat.new()
    heightmap_format.format = RenderingDevice.DATA_FORMAT_R8_UNORM
    heightmap_format.width = po2_dimensions
    heightmap_format.height = po2_dimensions
    heightmap_format.usage_bits = \
        RenderingDevice.TEXTURE_USAGE_STORAGE_BIT + \
        RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT + \
        RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT

    heightmap_rid = rd.texture_create(heightmap_format, RDTextureView.new())

    var heightmap_uniform: RDUniform = RDUniform.new()
    heightmap_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
    heightmap_uniform.binding = 0
    heightmap_uniform.add_id(heightmap_rid)

    var gradient_format: RDTextureFormat = RDTextureFormat.new()
    gradient_format.format = RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM
    gradient_format.width = gradient_tex.width
    gradient_format.height = 1
    gradient_format.usage_bits = RenderingDevice.TEXTURE_USAGE_STORAGE_BIT

    gradient_rid = rd.texture_create(gradient_format, RDTextureView.new(), [gradient_tex.get_image().get_data()])

    var gradient_uniform: RDUniform = RDUniform.new()
    gradient_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
    gradient_uniform.binding = 1
    gradient_uniform.add_id(gradient_rid)

    uniform_set = rd.uniform_set_create([heightmap_uniform, gradient_uniform], shader_rid, 0)
    pipeline = rd.compute_pipeline_create(shader_rid)

func compute_island_gpu(heightmap: Image) -> void:
    if rd == null:
        init_gpu()
    if rd == null:
        $CenterContainer/VBoxContainer/PanelContainer2/VBoxContainer/HBoxContainer2/Label2.text = "GPU unavailable"
        return

    rd.texture_update(heightmap_rid, 0, heightmap.get_data())

    var compute_list: int = rd.compute_list_begin()
    rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
    rd.compute_list_bind_uniform_set(compute_list, uniform_set, 0)
    var workgroup_x: int = ceil(float(po2_dimensions) / 4.0)  # Smaller workgroup for mobile
    var workgroup_y: int = ceil(float(po2_dimensions) / 4.0)
    rd.compute_list_dispatch(compute_list, workgroup_x, workgroup_y, 1)
    rd.compute_list_end()

    rd.submit()
    rd.sync()

    var output_bytes: PackedByteArray = rd.texture_get_data(heightmap_rid, 0)
    var island_img: Image = Image.create_from_data(po2_dimensions, po2_dimensions, false, Image.FORMAT_L8, output_bytes)
    display_island(island_img)

func cleanup_gpu() -> void:
    if rd == null:
        return
    rd.free_rid(pipeline)
    pipeline = RID()
    rd.free_rid(uniform_set)
    uniform_set = RID()
    rd.free_rid(gradient_rid)
    gradient_rid = RID()
    rd.free_rid(heightmap_rid)
    heightmap_rid = RID()
    rd.free_rid(shader_rid)
    shader_rid = RID()
    rd.free()
    rd = null

func load_shader(device: RenderingDevice, path: String) -> RID:
    var shader_file_data: RDShaderFile = load(path)
    var shader_spirv: RDShaderSPIRV = shader_file_data.get_spirv()
    return device.shader_create_from_spirv(shader_spirv)

func compute_island_cpu(heightmap: Image) -> void:
    var center: Vector2i = Vector2i(po2_dimensions / 2, po2_dimensions / 2)
    var max_dist: float = float(po2_dimensions) / 2.0  # Precompute max distance
    
    # Sample fewer pixels for mobile (step by 2)
    for y: int in range(0, po2_dimensions, 2):
        for x: int in range(0, po2_dimensions, 2):
            var coord: Vector2i = Vector2i(x, y)
            var pixel: Color = heightmap.get_pixelv(coord)
            var distance: float = Vector2(center).distance_to(Vector2(coord))
            var gradient_color: Color = gradient.sample(distance / max_dist)
            pixel.v *= gradient_color.v
            if pixel.v < 0.2:
                pixel.v = 0.0
            heightmap.set_pixelv(coord, pixel)
    
    # Quick interpolation for skipped pixels
    heightmap.resize(po2_dimensions, po2_dimensions, Image.INTERPOLATE_BILINEAR)
    display_island(heightmap)

func display_island(island: Image) -> void:
    island_rect.texture = ImageTexture.create_from_image(island)
    var stop_time: int = Time.get_ticks_usec()
    var elapsed: int = stop_time - start_time
    $CenterContainer/VBoxContainer/PanelContainer2/VBoxContainer/HBoxContainer/Label2.text = "%s ms" % str(elapsed * 0.001).pad_decimals(1)

func _on_random_button_pressed() -> void:
    randomize_seed()

func _on_create_button_gpu_pressed() -> void:
    var heightmap: Image = prepare_image()
    compute_island_gpu.call_deferred(heightmap)

func _on_create_button_cpu_pressed() -> void:
    var heightmap: Image = prepare_image()
    compute_island_cpu.call_deferred(heightmap)
