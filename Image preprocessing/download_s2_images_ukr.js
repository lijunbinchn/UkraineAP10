var UKR_shp = ee.FeatureCollection("projects/lijunbinchn/assets/ukr_parcel/gadm41_UKR_0");
 
// 封装生成时间段的函数
function generateDateRanges(year, periods) {
  // 初始化空的数组来存储日期范围
  var dateRanges = [];
  
  // 通过循环来生成时间段
  for (var i = 0; i < periods.length; i++) {
    var startMonth = periods[i][0];
    var endMonth = periods[i][1];
    
    // 创建开始和结束的日期
    var startDate = ee.Date.fromYMD(year, startMonth, 1);
    var endDate = ee.Date.fromYMD(year, endMonth, 1);
    
    // 如果结束月份是12月，则结束日期为下一年的1月1号
    if (endMonth === 12) {
      endDate = ee.Date.fromYMD(year + 1, 1, 1); // 下一年的1月1号
    } else {
      // 如果结束月份不是12月，则将结束日期设置为下个月的1号
      endDate = endDate.advance(1, 'month');
    }
    
    // 将日期范围添加到数组
    dateRanges.push([startDate, endDate]);
  }
  
  // 返回生成的日期范围
  return dateRanges;
}

// S2去云、去雪
function mask_s2_snow_clouds(image) {
  var scl = image.select('SCL');
  var Cloud_Shadows = 1 << 3;
  var Clouds_Low_Probability = 1 << 7;
  var	Clouds_Medium_Probability = 1 << 8;
  var Clouds_High_Probability = 1 << 9;
  var Cirrus = 1 << 10;
  var	Snow_Ice = 1 << 11;
  
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  
  var mask = scl.bitwiseAnd(Cloud_Shadows).eq(0)
    .and(scl.bitwiseAnd(Clouds_Low_Probability).eq(0))
    .and(scl.bitwiseAnd(Clouds_Medium_Probability).eq(0))
    .and(scl.bitwiseAnd(Clouds_High_Probability).eq(0))
    .and(scl.bitwiseAnd(Cirrus).eq(0))
    .and(scl.bitwiseAnd(Snow_Ice).eq(0))
    .and(qa.bitwiseAnd(cloudBitMask).eq(0))
    .and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  
  return image.updateMask(mask).set('system:time_start',image.get('system:time_start'));
}

function sentinel2MSI(img){
  return img.select(
                      ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9', 'B11','B12','QA60']
                      ,['aerosol', 'blue', 'green', 'red','red1','red2','red3','nir','red4','h2o','swir1', 'swir2','QA60']
                    )
                    // .divide(10000).toDouble()
                    .set('solar_azimuth',img.get('MEAN_SOLAR_AZIMUTH_ANGLE'))
                    .set('solar_zenith',img.get('MEAN_SOLAR_ZENITH_ANGLE') )
                    .set('system:time_start',img.get('system:time_start'));
}

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

var roi = UKR_shp
Map.centerObject(roi, 8)
Map.addLayer(roi, {color:"red"}, "UKR_shp")
print("UKR_shp",roi)
var country = "UKR"

// 选定的年份
var year = 2023;
// 要导出的时间段
var period = [[1,12]];
// 要导出的波段
var band_export = ['red', 'green', 'blue'];

// 生成月份后缀
// 直接使用 map 生成后缀
var month_suffixes = period.map(function(p) {
  return p[0] + "_" + p[1] + "_month";
});

// 打印结果
print('生成的后缀:', month_suffixes);

// 调用函数并打印结果
var dateRanges = generateDateRanges(year, period);
print('生成的时间段:', dateRanges);

// 选择需要的波段
var bands = ['red', 'green', 'blue'] // ,'NDWI','MNDWI','LSWI','MBWI','SIT','SI2','SI3','NDSI'

// 筛选影像
var startDate = dateRanges[0][0]
var endDate = dateRanges[0][1]
print(startDate, endDate)

var s2_col= ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                    .filterBounds(roi)
                    .filterDate(startDate,endDate)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))
                    .map(mask_s2_snow_clouds)
                    .map(sentinel2MSI)
                    .select(bands)
                    
print("s2_col", s2_col)
Map.addLayer(s2_col, { bands: ['red', 'green', 'blue'], min: 0,max: 10000,gamma: 1.4}, "s2_col");

// 进行中值合成
var s2_col_median = s2_col.median().clip(roi)
Map.addLayer(s2_col_median, { bands: ['red', 'green', 'blue'], min: 0,max: 10000,gamma: 1.4}, "s2_col_median");

// 设置导出的参数
var description = "s2_" + year + '_' + month_suffixes[0] + "_median_" + country
var file_name = description
print("export description:", description)
var folder = "s2_" + year + '_median_'+ country
var region = roi.geometry().bounds()
var scale = 10
var crs = 'EPSG:3857'
var no_data_value = 0;

// 导出中值合成影像
// 选择需要的波段，转为16位无符号整型，
var s2_col_median_uint16 = s2_col_median.toUint16()
print("s2_col_median_uint16",s2_col_median_uint16)
Map.addLayer(s2_col_median_uint16, { bands: ['red', 'green', 'blue'], min: 0,max: 10000,gamma: 1.4}, "s2_col_median_uint16_" + month_suffixes[0]);
export_image(s2_col_median_uint16, description, folder, region, scale, crs, file_name, no_data_value)





