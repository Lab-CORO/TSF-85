import omni.ui as ui

class CoRoPanel:
    """UI wrapper. Holds UI widgets, calls back into the Extension."""

    def __init__(self, ext):
        self._ext = ext
        self.window = None

        self.out_dir_label = None
        self.files_preview = None
        self.sensor_label = None
        self.mode_label = None
        self.def_nodes = None
        self.def_mesh = None
        self.def_dz_count = None
        self.def_dz_preview = None
        self.pred_preview = None
        self.show_heatmap = None
        self.heatmap_img = None
        self.log_label = None

        self.out_dir_field = None
        self.base_name_field = None

    def show(self):
        if self.window is None:
            self._build()
        self.window.visible = True


    def _build(self):
        ext = self._ext

        self.window = ui.Window("CoRo Tactile Sensor", width=550, height=400)
        with self.window.frame:
            with ui.VStack(spacing=0.3):

                with ui.CollapsableFrame("Output settings", collapsed=False, style={"margin": 3, "padding": 1}):
                    with ui.VStack(spacing=2):
                        self.out_dir_label = ui.Label(f"Directory: {ext._output_dir}")

                        with ui.HStack(spacing=2):
                            ui.Label("Output dir:", width=110)
                            self.out_dir_field = ui.StringField(width=270)
                            self.out_dir_field.model.set_value(str(ext._output_dir))
                            ui.Button("Apply dir", clicked_fn=ext._apply_output_dir)

                        with ui.HStack(spacing=2):
                            ui.Label("Default name:", width=110)
                            self.base_name_field = ui.StringField(width=270)
                            self.base_name_field.model.set_value(ext._base_name)
                            ui.Button("Apply name", clicked_fn=ext._apply_base_name)

                        ui.Button("New run", clicked_fn=ext._new_run)
                        self.files_preview = ui.Label(ext._files_preview_text())

                with ui.CollapsableFrame("Prim information", collapsed=False, style={"margin": 3, "padding": 1}):
                    with ui.VStack(spacing=2):
                        self.sensor_label = ui.Label("Active sensor: (none)")
                        self.mode_label = ui.Label("Select sensor once")
                        with ui.HStack(spacing=2):
                            ui.Button("Pick sensor", clicked_fn=ext._pick_sensor)
                            ui.Button("Clear", clicked_fn=ext._clear_sensor)

                with ui.CollapsableFrame("Deformable body information", collapsed=True, style={"margin": 3, "padding": 1}):
                    with ui.VStack(spacing=2):
                        self.def_nodes = ui.Label(f"Nodes in CSV: {len(ext._nodes_ids)}")
                        self.def_mesh = ui.Label("Mesh: (none)")
                        self.def_dz_count = ui.Label("dZ count: (none)")
                        self.def_dz_preview = ui.Label("dZ preview: (none)")
                        self.pred_preview = ui.Label("Prediction preview: (none)")


                with ui.CollapsableFrame("Tactile map visualization", collapsed=True, style={"margin": 3, "padding": 1}):
                    with ui.VStack(spacing=2):
                        with ui.HStack(spacing=2):
                            self.show_heatmap = ui.CheckBox()
                            self.show_heatmap.model.set_value(True)
                            ui.Label("Show prediction heatmap")

                        with ui.HStack(height=240):
                            ui.Spacer()
                            self.heatmap_img = ui.Image(width=460, height=240)
                            ui.Spacer()

                self.log_label = ui.Label("Logging: (starts when Play is pressed)")
