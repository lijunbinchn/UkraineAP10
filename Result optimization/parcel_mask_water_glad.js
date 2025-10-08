var UKR = ee.FeatureCollection("projects/lijunbinchn/assets/ukr_parcel/gadm41_UKR_0"),
    glad_global_water = ee.ImageCollection("projects/glad/water/C2/annual"),
    UKR_parcel_result = ee.Image("projects/lijunbinchn/assets/ukr_parcel/UKR_parcel_result_2023_HBGNet");

// UKR_parcel_result 替换为自己的模型推理的地块结果

var roi = UKR
Map.addLayer(UKR, {color:"red"}, "UKR")
Map.centerObject(roi, 5)

var ukr_parcel_2023 = UKR_parcel_result
Map.addLayer(ukr_parcel_2023, {palette:"green"}, "ukr_parcel_2023")
print(ukr_parcel_2023,"ukr_parcel_2023")

// 加载年度水体百分比集合
print("glad_global_water:", glad_global_water)
// 筛选2023年数据
var global_water_2023_percent = glad_global_water
  .filter(ee.Filter.eq('system:index', '2023')) // 匹配索引为"2023"的影像
  .first();
global_water_2023_percent = global_water_2023_percent.updateMask(global_water_2023_percent).clipToCollection(UKR)
print("global_water_2023_percent:", global_water_2023_percent)
Map.addLayer(global_water_2023_percent, {min:0, max:100, palette: ['white', 'blue']}, 'global_water_2023_percent');

// 使用 50% 的阈值提取水体分布
// 生成二值掩膜（1=水体，0=非水体）
var global_water_2023 = global_water_2023_percent.gt(50); // gte()表示"大于等于"
global_water_2023 = global_water_2023.updateMask(global_water_2023)
// 可选：将结果转为整型（便于导出）
global_water_2023 = global_water_2023.uint8();
print("global_water_2023:", global_water_2023)
Map.addLayer(global_water_2023, {palette: ['blue']}, 'global_water_2023');

// 使用 glad_water_extent 对 ukr_parcel进行掩膜
var masked_parcel_2023 = ee.Image(0);
masked_parcel_2023 = masked_parcel_2023.where(ukr_parcel_2023.eq(1),1)
masked_parcel_2023 = masked_parcel_2023.where(global_water_2023.eq(1),0)
masked_parcel_2023 = masked_parcel_2023.updateMask(masked_parcel_2023).clip(roi).uint8()
print("masked_parcel_2023:", masked_parcel_2023)
Map.addLayer(masked_parcel_2023,{palette:"green"},"masked_parcel_2023",false)

export_image(masked_parcel_2023, 
            "ukr_parcel_result_HBGNet_2023", 
            "ukr_parcel_result_HBGNet_2023", 
            roi.geometry().bounds(), 
            10, 
            'EPSG:3857', 
            "ukr_parcel_result_HBGNet_2023", 
            0)

// 导出影像
function export_image(image, description, folder, region, scale, crs, fileName, no_data_value){
    Export.image.toDrive({
      image:image,
      description:description,
      folder:folder,
      region:region,
      scale:scale,
      crs:crs,
      fileNamePrefix:fileName,
      maxPixels: 1e13,
      formatOptions: {
          noData: no_data_value
        }
    });
}
