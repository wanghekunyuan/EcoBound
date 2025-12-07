import arcpy
from Ecobound import align_rasters

def script_tool():
    template = arcpy.GetParameterAsText(0)  # Template raster
    value_table = arcpy.GetParameter(1)     # GPValueTable: [Input Raster, Continuous]
    output_folder = arcpy.GetParameterAsText(2)

    arcpy.env.overwriteOutput = True

    raster_list = []
    continuous_flags = []

    for row_index in range(value_table.rowCount):
        raster_path = value_table.getValue(row_index, 0)
        is_continuous = value_table.getValue(row_index, 1) in [True, "true", "True", 1]
        raster_list.append(raster_path)
        continuous_flags.append(is_continuous)

    def notify(msg):
        arcpy.AddMessage(msg)

    align_rasters(
        template=template,
        raster_list=raster_list,
        output_path=output_folder,
        continuous=continuous_flags,
        message_callback=notify
    )

if __name__ == "__main__":
    script_tool()
