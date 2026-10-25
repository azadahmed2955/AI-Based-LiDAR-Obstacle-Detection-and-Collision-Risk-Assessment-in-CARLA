#include <unordered_map>
#include <algorithm>
#include <atomic>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <carla/rpc/VehicleControl.h>
#include <string>
#include <carla/sensor/data/LidarMeasurement.h>
#include <iostream>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>

#include <vector>
#include <cmath>
#include <limits>
#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/Client.h>
#include <carla/client/Walker.h>
#include <carla/rpc/WalkerControl.h>
#include <carla/client/WalkerAIController.h>
#include <carla/client/Map.h>
#include <carla/client/Sensor.h>
#include <carla/client/TimeoutException.h>
#include <carla/client/World.h>
#include <carla/geom/Transform.h>
#include <carla/image/ImageIO.h>
#include <carla/image/ImageView.h>
#include <carla/sensor/data/Image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <open3d/Open3D.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

namespace cc = carla::client;
namespace cg = carla::geom;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

#define EXPECT_TRUE(pred) if (!(pred)) { throw std::runtime_error(#pred); }

/// Pick a random element from @a range.
template <typename RangeT, typename RNG>
static auto &RandomChoice(const RangeT &range, RNG &&generator) {
  EXPECT_TRUE(range.size() > 0u);
  std::uniform_int_distribution<size_t> dist{0u, range.size() - 1u};
  return range[dist(std::forward<RNG>(generator))];
}

/// Save a semantic segmentation image to disk converting to CityScapes palette.
/*
static void SaveSemSegImageToDisk(const csd::Image &image) {
  using namespace carla::image;

  char buffer[9u];
  std::snprintf(buffer, sizeof(buffer), "%08zu", image.GetFrame());
  auto filename = "_images/"s + buffer + ".png";

  auto view = ImageView::MakeColorConvertedView(
      ImageView::MakeView(image),
      ColorConverter::CityScapesPalette());
  ImageIO::WriteView(filename, view);
}
*/

static auto ParseArguments(int argc, const char *argv[]) {
  EXPECT_TRUE((argc == 1u) || (argc == 3u));
  using ResultType = std::tuple<std::string, uint16_t>;
  return argc == 3u ?
      ResultType{argv[1u], std::stoi(argv[2u])} :
      ResultType{"localhost", 2000u};
}

static Eigen::Vector3d GetClusterColor(size_t cluster_id) {
  static const std::vector<Eigen::Vector3d> palette = {
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {0.0, 0.0, 1.0},
      {1.0, 1.0, 0.0},
      {1.0, 0.0, 1.0},
      {0.0, 1.0, 1.0},
      {1.0, 0.5, 0.0},
      {0.5, 1.0, 0.0},
      {0.5, 0.0, 1.0},
      {1.0, 0.5, 0.5}
  };

  return palette[
      cluster_id %
      palette.size()];
}

int main(int argc, const char *argv[]) {
  try {

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "carla_ai");
    Ort::SessionOptions session_options;

    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    const char *pfe_model_path =
        "/home/ageda_user/carla_ai/pointpillars/"
        "pointpillars_int8/model/pp_pfe.onnx";

    Ort::Session pfe_session(
        env,
        pfe_model_path,
        session_options);

    const char *rpn_model_path =
        "/home/ageda_user/carla_ai/pointpillars/"
        "pointpillars_int8/model/pp_rpn.onnx";

    Ort::Session rpn_session(
        env,
        rpn_model_path,
        session_options);

    std::cout
        << "ONNX Runtime initialized successfully."
        << std::endl;

    std::cout
        << "PointPillars PFE model loaded successfully."
        << std::endl;

    std::cout
        << "PointPillars RPN model loaded successfully."
        << std::endl;

    std::string host;
    uint16_t port;
    std::tie(host, port) = ParseArguments(argc, argv);

    std::mt19937_64 rng((std::random_device())());

    auto client = cc::Client(host, port);
    client.SetTimeout(40s);

    std::cout << "Client API version : " << client.GetClientVersion() << '\n';
    std::cout << "Server API version : " << client.GetServerVersion() << '\n';

    // Load town
    auto world = client.LoadWorld("Town05_Opt");

    // ---------------------------------------------------------
    // Deterministic synchronous CARLA experiment.
    //
    // 0.1 s simulation step = 10 Hz, matching LiDAR.
    // CARLA will advance only when world.Tick() is called.
    // ---------------------------------------------------------
    auto original_world_settings =
        world.GetSettings();

    auto sync_settings =
        original_world_settings;

    sync_settings.synchronous_mode =
        true;

    sync_settings.fixed_delta_seconds =
        0.1;

    world.ApplySettings(
        sync_settings,
        2s);

    std::cout
        << "CARLA synchronous mode enabled"
        << " | fixed delta = 0.1 s"
        << std::endl;

    // ---------------------------------------------------------
    // Deterministic controlled experiment.
    // Use the same ego vehicle and same spawn point every run.
    // ---------------------------------------------------------
    auto blueprint_library = world.GetBlueprintLibrary();

    auto ego_bp_ptr =
        blueprint_library->Find("vehicle.tesla.model3");

    EXPECT_TRUE(ego_bp_ptr != nullptr);

    auto blueprint =
        *ego_bp_ptr;

    auto map = world.GetMap();

    auto spawn_points =
        map->GetRecommendedSpawnPoints();

    EXPECT_TRUE(spawn_points.size() > 0u);

    // Fixed spawn point for repeatable experiments.
    const size_t ego_spawn_index = 0u;

    EXPECT_TRUE(
        ego_spawn_index <
        spawn_points.size());

    auto transform =
        spawn_points[ego_spawn_index];

    std::cout
        << "Fixed ego: Tesla Model 3"
        << " | Spawn index: "
        << ego_spawn_index
        << std::endl;

    // Spawn the vehicle.
    auto actor = world.SpawnActor(blueprint, transform);
    std::cout << "Spawned " << actor->GetDisplayId() << '\n';

    auto vehicle =
        boost::static_pointer_cast<cc::Vehicle>(actor);

    // ---------------------------------------------------------
    // In synchronous mode, commit the ego spawn before using
    // vehicle->GetTransform() for waypoint calculations.
    // ---------------------------------------------------------
    const uint64_t ego_setup_frame =
        world.Tick(2s);

    const auto committed_ego_tf =
        vehicle->GetTransform();

    std::cout
        << "Ego spawn committed"
        << " | Frame: "
        << ego_setup_frame
        << " | X: "
        << committed_ego_tf.location.x
        << " | Y: "
        << committed_ego_tf.location.y
        << " | Z: "
        << committed_ego_tf.location.z
        << std::endl;

    // Apply controlled C++ throttle for TTC experiment.
    vehicle->SetAutopilot(false);

    carla::rpc::VehicleControl ego_control;
    ego_control.throttle = 0.25f;
    ego_control.steer = 0.0f;
    ego_control.brake = 0.0f;

    vehicle->ApplyControl(ego_control);

    std::cout
        << "Controlled ego throttle enabled: 0.25"
        << std::endl;

    // ---------------------------------------------------------
    // Controlled TTC experiment:
    // Spawn one stationary vehicle about 15 m ahead of ego.
    // ---------------------------------------------------------
    auto target_bp =
        blueprint_library->Find("vehicle.tesla.model3");

    EXPECT_TRUE(target_bp != nullptr);

    // ---------------------------------------------------------
    // Spawn stationary target about 20 m ahead using the
    // actual CARLA driving lane.
    //
    // This avoids placing the target off-road or inside
    // roadside geometry.
    // ---------------------------------------------------------
    const auto ego_transform =
        vehicle->GetTransform();

    auto ego_waypoint =
        map->GetWaypoint(
            ego_transform.location);

    if (ego_waypoint == nullptr) {
      throw std::runtime_error(
          "Could not find driving waypoint for ego vehicle.");
    }

    const auto ego_wp_tf =
        ego_waypoint->GetTransform();

    std::cout
        << "EGO WAYPOINT"
        << " | X: " << ego_wp_tf.location.x
        << " | Y: " << ego_wp_tf.location.y
        << " | Z: " << ego_wp_tf.location.z
        << std::endl;

    auto next_5m =
        ego_waypoint->GetNext(5.0);

    std::cout
        << "GetNext(5m) candidates: "
        << next_5m.size()
        << std::endl;

    for (size_t i = 0; i < next_5m.size(); ++i) {
      const auto tf =
          next_5m[i]->GetTransform();

      std::cout
          << " NEXT5[" << i << "]"
          << " X: " << tf.location.x
          << " Y: " << tf.location.y
          << " Z: " << tf.location.z
          << std::endl;
    }

    auto target_waypoints =
        ego_waypoint->GetNext(20.0);

    std::cout
        << "GetNext(20m) candidates: "
        << target_waypoints.size()
        << std::endl;

    for (size_t i = 0;
         i < target_waypoints.size();
         ++i) {

      const auto tf =
          target_waypoints[i]->GetTransform();

      const float dx =
          tf.location.x -
          ego_transform.location.x;

      const float dy =
          tf.location.y -
          ego_transform.location.y;

      const float straight_distance =
          std::sqrt(
              dx * dx +
              dy * dy);

      std::cout
          << " NEXT20[" << i << "]"
          << " X: " << tf.location.x
          << " Y: " << tf.location.y
          << " Z: " << tf.location.z
          << " | EuclideanFromEgo: "
          << straight_distance
          << " m"
          << std::endl;
    }

    if (target_waypoints.empty()) {
      throw std::runtime_error(
          "Could not find waypoint 20 m ahead.");
    }

    // For this controlled experiment use the first
    // continuation of the current driving lane.
    auto target_waypoint =
        target_waypoints.front();

    auto target_transform =
        target_waypoint->GetTransform();

    // Small vertical offset helps avoid spawn collision
    // with the road surface.
    // Keep legacy GT vehicle outside the active scenario.
    // Older evaluation code still references target_vehicle,
    // so keep it alive but far above the scene.
    target_transform.location.z +=
        100.0f;

    auto target_actor =
        world.TrySpawnActor(
            *target_bp,
            target_transform);

    if (target_actor == nullptr) {
      throw std::runtime_error(
          "Could not spawn target at 20 m road waypoint.");
    }

    auto target_vehicle =
        boost::static_pointer_cast<cc::Vehicle>(
            target_actor);

    target_vehicle->SetAutopilot(false);
    target_vehicle->SetSimulatePhysics(false);

    // Commit setup in synchronous mode.
    const uint64_t setup_frame =
        world.Tick(2s);

    const auto confirmed_target =
        target_vehicle->GetTransform().location;

    const auto confirmed_ego =
        vehicle->GetTransform().location;

    const float setup_dx =
        confirmed_target.x -
        confirmed_ego.x;

    const float setup_dy =
        confirmed_target.y -
        confirmed_ego.y;

    const float setup_distance =
        std::sqrt(
            setup_dx * setup_dx +
            setup_dy * setup_dy);

    std::cout
        << "Stationary target spawned on road waypoint."
        << " | Setup frame: "
        << setup_frame
        << " | EgoX: "
        << confirmed_ego.x
        << " | EgoY: "
        << confirmed_ego.y
        << " | TargetX: "
        << confirmed_target.x
        << " | TargetY: "
        << confirmed_target.y
        << " | Separation: "
        << setup_distance
        << " m"
        << std::endl;


    // =========================================================
    // CONTROLLED PEDESTRIAN CROSSING SCENARIO
    //
    // Exactly:
    //   1 ego vehicle
    //   1 crossing pedestrian
    //
    // No random cars, cyclists or pedestrians.
    // =========================================================

    std::vector<boost::shared_ptr<cc::Actor>>
        scenario_actors;

    // Kept empty so existing cleanup code remains valid.
    std::vector<
        boost::shared_ptr<cc::WalkerAIController>>
        walker_controllers;

    // Find a road location approximately 15 m ahead.
    auto crossing_waypoints =
        ego_waypoint->GetNext(25.0);

    if (crossing_waypoints.empty()) {
      throw std::runtime_error(
          "Could not find pedestrian crossing waypoint.");
    }

    auto crossing_waypoint =
        crossing_waypoints.front();

    auto crossing_tf =
        crossing_waypoint->GetTransform();

    const auto crossing_right =
        crossing_tf.GetRightVector();

    // ---------------------------------------------------------
    // Spawn pedestrian on the RIGHT side of the road.
    // It will walk toward the LEFT side.
    // ---------------------------------------------------------
    auto walker_blueprints =
        blueprint_library->Filter(
            "walker.pedestrian.*");

    if (walker_blueprints->empty()) {
      throw std::runtime_error(
          "No pedestrian blueprints available.");
    }

    boost::shared_ptr<cc::Actor>
        crossing_walker_actor = nullptr;

    cg::Transform walker_tf =
        crossing_tf;

    float successful_pedestrian_offset =
        0.0f;

    // Try several positions on the right side of the road.
    // Some exact sidewalk locations may collide with map
    // geometry such as curbs, fences, trees or poles.
    const float pedestrian_offsets[] = {
        2.5f,
        3.0f,
        3.5f,
        4.0f,
        4.5f,
        5.0f
    };

    for (float offset :
         pedestrian_offsets) {

      walker_tf =
          crossing_tf;

      walker_tf.location.x +=
          offset *
          crossing_right.x;

      walker_tf.location.y +=
          offset *
          crossing_right.y;

      // Slightly above ground to avoid initial collision.
      walker_tf.location.z +=
          1.0f;

      for (const auto &walker_bp :
           *walker_blueprints) {

        crossing_walker_actor =
            world.TrySpawnActor(
                walker_bp,
                walker_tf);

        if (crossing_walker_actor != nullptr) {

          successful_pedestrian_offset =
              offset;

          break;
        }
      }

      if (crossing_walker_actor != nullptr) {
        break;
      }
    }

    if (crossing_walker_actor == nullptr) {
      throw std::runtime_error(
          "Could not spawn crossing pedestrian at any tested roadside position.");
    }

    std::cout
        << "Pedestrian spawn successful"
        << " | Lateral offset: "
        << successful_pedestrian_offset
        << " m"
        << " | X: "
        << walker_tf.location.x
        << " | Y: "
        << walker_tf.location.y
        << std::endl;

    auto crossing_walker =
        boost::static_pointer_cast<
            cc::Walker>(
                crossing_walker_actor);

    scenario_actors.push_back(
        crossing_walker_actor);

    world.Tick(2s);

    std::cout
        << "Controlled pedestrian spawned"
        << " | approximately 25 m ahead"
        << std::endl;

    // ---------------------------------------------------------
    // Pedestrian walking direction:
    // from right side -> left side of road.
    // ---------------------------------------------------------
    carla::rpc::WalkerControl
        pedestrian_control;

    pedestrian_control.direction =
        cg::Vector3D{
            -crossing_right.x,
            -crossing_right.y,
            0.0f};

    // Pedestrian initially waits at the roadside.
    pedestrian_control.speed =
        0.0f;

    pedestrian_control.jump =
        false;

    crossing_walker->ApplyControl(
        pedestrian_control);

    std::cout
        << "Pedestrian waiting at roadside."
        << std::endl;


    // Move spectator so we can see the vehicle from the simulator window.
    auto spectator = world.GetSpectator();
    transform.location += 32.0f * transform.GetForwardVector();
    transform.location.z += 2.0f;
    transform.rotation.yaw += 180.0f;
    transform.rotation.pitch = -15.0f;
    spectator->SetTransform(transform);

    // LiDAR sensor for the project.
  auto lidar_bp_ptr = blueprint_library->Find("sensor.lidar.ray_cast");
EXPECT_TRUE(lidar_bp_ptr != nullptr);

auto lidar_bp = *lidar_bp_ptr;

lidar_bp.SetAttribute("channels", "64");
lidar_bp.SetAttribute("range", "50");
lidar_bp.SetAttribute("rotation_frequency", "10");
lidar_bp.SetAttribute("points_per_second", "300000");
lidar_bp.SetAttribute("upper_fov", "10");
lidar_bp.SetAttribute("lower_fov", "-30");

    auto lidar_transform = cg::Transform{
        cg::Location{0.0f, 0.0f, 2.5f},
        cg::Rotation{0.0f, 0.0f, 0.0f}};

    auto lidar_actor =
        world.SpawnActor(lidar_bp, lidar_transform, actor.get());

    auto lidar =
        boost::static_pointer_cast<cc::Sensor>(lidar_actor);

    std::cout << "LiDAR spawned." << std::endl;

    std::ofstream csv("lidar_results.csv");
    csv << "frame,raw_points,roi_points,non_ground_points,clusters,nearest_obstacle,track_x,track_y,track_distance,smoothed_distance,closing_speed,ttc,raw_risk,stable_risk,ped_gt_x,ped_gt_y,ped_gt_distance,distance_error,ped_surface_x,ped_surface_y,ped_surface_distance,surface_error,processing_ms,processing_fps,ai_track_valid,ai_class,ai_confidence,ai_x,ai_y,ai_distance,ai_smoothed_distance,ai_closing_speed,ai_ttc,ai_raw_risk,ai_stable_risk,ai_center_error,ai_distance_error,ego_speed,pedestrian_hazard,obstacle_hazard,control_state\n";

    std::mutex lidar_processing_mutex;
    std::condition_variable lidar_processing_cv;

    uint64_t last_processed_lidar_frame = 0;

    // Shared signal from PointPillars to ego-vehicle control.
    std::atomic<bool> pedestrian_hazard{false};

    // Independent classical LiDAR safety signal.
    // Stops ego for ANY physical obstacle directly ahead.
    std::atomic<bool> obstacle_hazard{false};

    std::mutex vis_points_mutex;
    std::vector<Eigen::Vector3d> latest_vis_points;
    std::vector<Eigen::Vector3d> latest_vis_colors;
    std::atomic<bool> vis_running{true};

    std::thread vis_thread(
        [&vis_points_mutex,
         &latest_vis_points,
         &latest_vis_colors,
         &vis_running]() {

      open3d::visualization::Visualizer vis;

      vis.CreateVisualizerWindow(
          "CARLA LiDAR",
          1280,
          720);

      auto cloud =
          std::make_shared<
              open3d::geometry::PointCloud>();

      // ---------------------------------------------------
      // Create a ground grid similar to a LiDAR viewer.
      // LiDAR is mounted 2.5 m above the road, therefore
      // the road appears close to z = -2.5 in sensor frame.
      // ---------------------------------------------------
      auto grid =
          std::make_shared<
              open3d::geometry::LineSet>();

      int grid_index = 0;

      const double grid_z = -2.5;
      const double x_min = -10.0;
      const double x_max = 50.0;
      const double y_min = -30.0;
      const double y_max = 30.0;
      const double grid_step = 2.0;

      for (double x = x_min;
           x <= x_max;
           x += grid_step) {

        grid->points_.push_back(
            Eigen::Vector3d(
                x, y_min, grid_z));

        grid->points_.push_back(
            Eigen::Vector3d(
                x, y_max, grid_z));

        grid->lines_.push_back(
            Eigen::Vector2i(
                grid_index,
                grid_index + 1));

        grid->colors_.push_back(
            Eigen::Vector3d(
                0.30, 0.30, 0.30));

        grid_index += 2;
      }

      for (double y = y_min;
           y <= y_max;
           y += grid_step) {

        grid->points_.push_back(
            Eigen::Vector3d(
                x_min, y, grid_z));

        grid->points_.push_back(
            Eigen::Vector3d(
                x_max, y, grid_z));

        grid->lines_.push_back(
            Eigen::Vector2i(
                grid_index,
                grid_index + 1));

        grid->colors_.push_back(
            Eigen::Vector3d(
                0.30, 0.30, 0.30));

        grid_index += 2;
      }

      vis.AddGeometry(grid);

      bool cloud_added = false;

      auto &render_option =
          vis.GetRenderOption();

      render_option.background_color_ =
          Eigen::Vector3d(
              0.0, 0.0, 0.0);

      render_option.point_size_ =
          3.0;

      render_option.line_width_ =
          1.0;

      render_option.show_coordinate_frame_ =
          true;

      while (vis_running) {

        std::vector<Eigen::Vector3d>
            local_points;

        std::vector<Eigen::Vector3d>
            local_colors;

        {
          std::lock_guard<std::mutex> lock(
              vis_points_mutex);

          local_points =
              latest_vis_points;

          local_colors =
              latest_vis_colors;
        }

        if (!local_points.empty()) {

          cloud->points_ =
              local_points;

          cloud->colors_ =
              local_colors;

          if (!cloud_added) {

            vis.AddGeometry(
                cloud,
                false);

            cloud_added = true;

            auto &view =
                vis.GetViewControl();

            // Oblique 3D view similar to a typical
            // automotive LiDAR viewer.
            view.SetLookat(
                Eigen::Vector3d(
                    12.0, 0.0, -1.5));

            view.SetFront(
                Eigen::Vector3d(
                    -0.75, -0.35, 0.55));

            view.SetUp(
                Eigen::Vector3d(
                    0.0, 0.0, 1.0));

            view.SetZoom(
                0.30);

          } else {

            vis.UpdateGeometry(
                cloud);
          }
        }

        if (!vis.PollEvents()) {

          vis_running =
              false;

          break;
        }

        vis.UpdateRender();

        std::this_thread::sleep_for(
            std::chrono::milliseconds(
                30));
      }

      vis.DestroyVisualizerWindow();
    });

    lidar->Listen(
        [&csv, vehicle, crossing_walker,
         &pfe_session, &rpn_session,
         &lidar_processing_mutex,
         &lidar_processing_cv,
         &last_processed_lidar_frame,
         &pedestrian_hazard,
         &obstacle_hazard,
         &vis_points_mutex,
         &latest_vis_points,
         &latest_vis_colors](auto data) {
      auto measurement =
          boost::static_pointer_cast<csd::LidarMeasurement>(data);

      std::vector<Eigen::Vector3d>
          vis_points;

      std::vector<Eigen::Vector3d>
          vis_colors;

      // PCL cluster visualization buffers.
      std::vector<Eigen::Vector3d>
          cluster_vis_points;

      std::vector<Eigen::Vector3d>
          cluster_vis_colors;

      vis_points.reserve(
          measurement->size());

      vis_colors.reserve(
          measurement->size());

      for (const auto &pt :
           *measurement) {

        const double x =
            static_cast<double>(
                pt.point.x);

        const double y =
            static_cast<double>(
                pt.point.y);

        const double z =
            static_cast<double>(
                pt.point.z);

        const double distance =
            std::sqrt(
                x * x +
                y * y +
                z * z);

        if (distance < 0.5 ||
            distance > 50.0) {
          continue;
        }

        // Visualization ROI:
        // keep mainly road users and nearby road area.
        // This removes most building/background points.
        if (x < 0.0 || x > 50.0 ||
            y < -8.0 || y > 8.0 ||
            z < -2.8 || z > 2.5) {
          continue;
        }

        vis_points.emplace_back(
            x, y, z);

        // -----------------------------------------------
        // Rainbow-style distance coloring.
        //
        // Near    -> red/orange
        // Medium  -> yellow/green
        // Far     -> cyan/blue/purple
        // -----------------------------------------------
        const double t =
            std::max(
                0.0,
                std::min(
                    1.0,
                    distance / 50.0));

        double r = 0.0;
        double g = 0.0;
        double b = 0.0;

        if (t < 0.25) {

          r = 1.0;
          g = t / 0.25;
          b = 0.0;

        } else if (t < 0.50) {

          r =
              1.0 -
              (t - 0.25) /
              0.25;

          g = 1.0;
          b = 0.0;

        } else if (t < 0.75) {

          r = 0.0;
          g = 1.0;

          b =
              (t - 0.50) /
              0.25;

        } else {

          r =
              (t - 0.75) /
              0.25;

          g =
              1.0 -
              (t - 0.75) /
              0.25;

          b = 1.0;
        }

        vis_colors.emplace_back(
            r, g, b);
      }

      // -----------------------------------------------------
      // Capture CARLA ground truth immediately when this
      // LiDAR callback begins.
      //
      // AI processing takes hundreds of milliseconds, so
      // querying vehicle transforms after inference would
      // compare an old LiDAR frame with a newer vehicle pose.
      // -----------------------------------------------------
      const auto lidar_frame_ego_tf =
          vehicle->GetTransform();

      // Pedestrian ground truth captured at the same
      // simulation frame as the LiDAR measurement.
      const auto lidar_frame_target_tf =
          crossing_walker->GetTransform();

      // Measure the actual ego-vehicle speed from CARLA.
      auto ego_velocity = vehicle->GetVelocity();

      float ego_speed = std::sqrt(
          ego_velocity.x * ego_velocity.x +
          ego_velocity.y * ego_velocity.y +
          ego_velocity.z * ego_velocity.z);

      std::cout << "EgoSpeed: "
                << ego_speed
                << " m/s | ";

      const auto processing_start =
          std::chrono::steady_clock::now();

      size_t roi_points = 0;
      size_t non_ground_points = 0;

      struct Point3D {
        float x;
        float y;
        float z;
      };

      // Raw LiDAR point format for the future AI detector.
      struct AIPoint {
        float x;
        float y;
        float z;
        float intensity;
      };

      std::vector<Point3D> obstacle_points;

      // Wider point set only for clustering research/visualization.
      // This does NOT replace the narrow TTC/risk ROI.
      std::vector<Point3D> clustering_points;

      std::vector<AIPoint> ai_points;
      std::vector<AIPoint> ai_kitti_points;

      // Raw-LiDAR emergency corridor.
      bool obstacle_in_driving_corridor =
          false;

      for (const auto &point : *measurement) {
        const float x = point.point.x;
        const float y = point.point.y;
        const float z = point.point.z;
        const float intensity = point.intensity;

        // ---------------------------------------------------
        // Generic obstacle safety corridor.
        //
        // Ignore ego body and road.
        // Stop for objects roughly:
        //   3-10 m ahead
        //   +/-1.6 m around lane centre
        // ---------------------------------------------------
        if (x > 3.0f &&
            x < 10.0f &&
            std::fabs(y) < 1.6f &&
            z > -1.3f &&
            z < 2.5f) {

          obstacle_in_driving_corridor =
              true;
        }

        // Keep only road-relevant LiDAR points.
        // This removes most building/background clutter.
        if (x >= 0.0f && x <= 50.0f &&
            y >= -12.0f && y <= 12.0f &&
            z >= -2.8f && z <= 3.0f) {

          ai_points.push_back(
              {x, y, z, intensity});
        }

        // Convert CARLA LiDAR coordinates to KITTI-style convention.
        // CARLA: x forward, y right, z up
        // KITTI-style: x forward, y left, z up
        const float kitti_x = x;
        const float kitti_y = -y;
        const float kitti_z = z;

        if (kitti_x >= 0.0f && kitti_x <= 50.0f &&
            kitti_y >= -12.0f && kitti_y <= 12.0f &&
            kitti_z >= -2.8f && kitti_z <= 1.0f) {
          ai_kitti_points.push_back(
              {kitti_x, kitti_y, kitti_z, intensity});
        }

        // ---------------------------------------------------
        // Wide clustering ROI for PCL visualization/research.
        // Keep more road-relevant objects around the ego.
        // ---------------------------------------------------
        if (x > 0.0f && x < 50.0f &&
            y > -12.0f && y < 12.0f &&
            z > -1.3f && z < 3.0f) {

          const float clustering_planar_distance =
              std::sqrt(x * x + y * y);

          if (clustering_planar_distance > 3.0f) {
            clustering_points.push_back({x, y, z});
          }
        }

        // ---------------------------------------------------
        // Existing narrow ROI for TTC / collision-risk logic.
        // Keep this unchanged.
        // ---------------------------------------------------
        if (x > 0.0f && x < 40.0f &&
            y > -2.5f && y < 2.5f &&
            z > -2.0f && z < 3.0f) {

          ++roi_points;

          if (z > -1.3f) {
            float planar_distance = std::sqrt(x * x + y * y);

            if (planar_distance > 3.0f) {
              ++non_ground_points;
              obstacle_points.push_back({x, y, z});
            }
          }
        }
      }

      // =====================================================
      // OpenCV LiDAR BEV visualization
      // =====================================================

      const int opencv_bev_width = 900;
      const int opencv_bev_height = 900;

      cv::Mat bev(
          opencv_bev_height,
          opencv_bev_width,
          CV_8UC3,
          cv::Scalar(8, 8, 8));

      // Display range:
      // forward: 0 ... 50 m
      // lateral: -25 ... +25 m
      const float bev_forward_max = 50.0f;
      const float bev_lateral_half = 25.0f;

      const float pixels_per_meter =
          static_cast<float>(opencv_bev_height) /
          bev_forward_max;

      // -----------------------------------------------------
      // Grid every 5 m.
      // -----------------------------------------------------
      for (int distance = 0;
           distance <= 50;
           distance += 5) {

        const int py =
            opencv_bev_height -
            static_cast<int>(
                distance *
                pixels_per_meter);

        cv::line(
            bev,
            cv::Point(0, py),
            cv::Point(opencv_bev_width - 1, py),
            cv::Scalar(45, 45, 45),
            1);

        if (distance > 0 && py >= 15) {
          cv::putText(
              bev,
              std::to_string(distance) + " m",
              cv::Point(10, py - 5),
              cv::FONT_HERSHEY_SIMPLEX,
              0.45,
              cv::Scalar(160, 160, 160),
              1);
        }
      }

      for (int lateral = -25;
           lateral <= 25;
           lateral += 5) {

        const int px =
            static_cast<int>(
                (lateral +
                 bev_lateral_half) *
                (opencv_bev_width /
                 (2.0f *
                  bev_lateral_half)));

        cv::line(
            bev,
            cv::Point(px, 0),
            cv::Point(px, opencv_bev_height - 1),
            cv::Scalar(45, 45, 45),
            1);
      }

      // -----------------------------------------------------
      // Draw LiDAR points.
      //
      // CARLA:
      // x = forward
      // y = right
      //
      // BEV:
      // forward = upward
      // left/right = horizontal
      // -----------------------------------------------------
      for (const auto &p : ai_points) {

        if (p.x < 0.0f ||
            p.x > bev_forward_max ||
            p.y < -bev_lateral_half ||
            p.y > bev_lateral_half) {
          continue;
        }

        const int px =
            static_cast<int>(
                opencv_bev_width * 0.5f +
                p.y *
                pixels_per_meter);

        const int py =
            static_cast<int>(
                opencv_bev_height -
                p.x *
                pixels_per_meter);

        if (px < 0 || px >= opencv_bev_width ||
            py < 0 || py >= opencv_bev_height) {
          continue;
        }

        // Height-based color.
        const float z_normalized =
            std::max(
                0.0f,
                std::min(
                    1.0f,
                    (p.z + 3.0f) / 5.0f));

        const int hue =
            static_cast<int>(
                (1.0f - z_normalized) *
                120.0f);

        cv::Mat hsv_pixel(
            1, 1, CV_8UC3,
            cv::Scalar(
                hue,
                255,
                255));

        cv::Mat bgr_pixel;

        cv::cvtColor(
            hsv_pixel,
            bgr_pixel,
            cv::COLOR_HSV2BGR);

        const cv::Vec3b color =
            bgr_pixel.at<cv::Vec3b>(0, 0);

        cv::circle(
            bev,
            cv::Point(px, py),
            1,
            cv::Scalar(
                color[0],
                color[1],
                color[2]),
            -1);
      }

      // -----------------------------------------------------
      // Ego vehicle marker.
      // -----------------------------------------------------
      const cv::Point ego_point(
          opencv_bev_width / 2,
          opencv_bev_height - 10);

      cv::circle(
          bev,
          ego_point,
          7,
          cv::Scalar(255, 255, 255),
          -1);

      cv::putText(
          bev,
          "EGO",
          cv::Point(
              ego_point.x + 10,
              ego_point.y - 5),
          cv::FONT_HERSHEY_SIMPLEX,
          0.5,
          cv::Scalar(255, 255, 255),
          1);

      cv::putText(
          bev,
          "CARLA LiDAR - Bird's Eye View",
          cv::Point(20, 30),
          cv::FONT_HERSHEY_SIMPLEX,
          0.75,
          cv::Scalar(255, 255, 255),
          2);

      cv::imshow(
          "CARLA LiDAR OpenCV BEV",
          bev);

      cv::waitKey(1);

      obstacle_hazard.store(
          obstacle_in_driving_corridor);

      // -----------------------------------------------------
      // PointPillars-style pillarization.
      // Spatial range:
      //   X: 0.00 ... 69.12 m
      //   Y: -39.68 ... 39.68 m
      // Pillar size:
      //   0.16 m x 0.16 m
      // Maximum:
      //   32 LiDAR points stored per pillar.
      // -----------------------------------------------------

      constexpr float voxel_x = 0.16f;
      constexpr float voxel_y = 0.16f;
      constexpr float min_x = 0.0f;
      constexpr float min_y = -39.68f;

      constexpr int grid_x = 432;
      constexpr int grid_y = 496;
      constexpr size_t max_points_per_pillar = 32;

      struct Pillar {
        int x_index;
        int y_index;
        std::vector<AIPoint> points;
      };

      std::unordered_map<long long, Pillar> pillar_map;

      for (const auto &p : ai_kitti_points) {

        const int px =
            static_cast<int>(
                std::floor((p.x - min_x) / voxel_x));

        const int py =
            static_cast<int>(
                std::floor((p.y - min_y) / voxel_y));

        if (px < 0 || px >= grid_x ||
            py < 0 || py >= grid_y) {
          continue;
        }

        const long long key =
            static_cast<long long>(py) * grid_x + px;

        auto it = pillar_map.find(key);

        if (it == pillar_map.end()) {
          Pillar pillar;
          pillar.x_index = px;
          pillar.y_index = py;
          pillar.points.reserve(max_points_per_pillar);

          it = pillar_map.emplace(
              key, std::move(pillar)).first;
        }

        if (it->second.points.size() <
            max_points_per_pillar) {

          it->second.points.push_back(p);
        }
      }

      size_t pillar_point_count = 0;

      for (const auto &entry : pillar_map) {
        pillar_point_count += entry.second.points.size();
      }

      // -----------------------------------------------------
      // Build PointPillars PFE input tensor.
      //
      // ONNX input shape:
      //   [20000, 32, 10]
      //
      // Layout:
      //   [pillar][point][feature]
      //
      // 10 features:
      //   0: x
      //   1: y
      //   2: z
      //   3: intensity
      //   4: x - mean_x
      //   5: y - mean_y
      //   6: z - mean_z
      //   7: x - pillar_center_x
      //   8: y - pillar_center_y
      //   9: z - pillar_center_z
      // -----------------------------------------------------

      constexpr size_t pfe_max_pillars = 20000;
      constexpr size_t pfe_max_points = 32;
      constexpr size_t pfe_features = 10;

      const size_t active_pillars =
          std::min(
              pillar_map.size(),
              pfe_max_pillars);

      // Zero initialization automatically pads:
      //   unused points in a pillar
      //   unused pillars up to 20000.
      std::vector<float> pfe_input(
          pfe_max_pillars *
          pfe_max_points *
          pfe_features,
          0.0f);

      // Keep pillar coordinates in exactly the same order
      // as the PFE input. These will be needed later
      // when scattering the 64 PFE features into BEV.
      std::vector<int> pfe_x_indices;
      std::vector<int> pfe_y_indices;

      pfe_x_indices.reserve(active_pillars);
      pfe_y_indices.reserve(active_pillars);

      // -----------------------------------------------------
      // PFE input-layout experiment.
      //
      // false = Layout A:
      //         [pillar][point][feature]
      //         matches ONNX shape [20000,32,10]
      //
      // true  = Layout B:
      //         repository CUDA-style flattened memory
      //         [pillar][feature][point]
      //
      // Layout B is diagnostic only. The ONNX tensor shape
      // remains [20000,32,10].
      // -----------------------------------------------------
      constexpr bool pfe_cuda_style_layout = false;

      static bool pfe_layout_printed = false;

      if (!pfe_layout_printed) {
        std::cout
            << "PFE input layout: "
            << (pfe_cuda_style_layout
                    ? "B = CUDA-style [pillar][feature][point]"
                    : "A = ONNX-style [pillar][point][feature]")
            << std::endl;

        pfe_layout_printed = true;
      }

      size_t pillar_index = 0;

      for (const auto &entry : pillar_map) {

        if (pillar_index >= pfe_max_pillars) {
          break;
        }

        const Pillar &pillar = entry.second;

        if (pillar.points.empty()) {
          continue;
        }

        float mean_x = 0.0f;
        float mean_y = 0.0f;
        float mean_z = 0.0f;

        for (const auto &pt : pillar.points) {
          mean_x += pt.x;
          mean_y += pt.y;
          mean_z += pt.z;
        }

        const float inv_count =
            1.0f /
            static_cast<float>(pillar.points.size());

        mean_x *= inv_count;
        mean_y *= inv_count;
        mean_z *= inv_count;

        const float center_x =
            min_x +
            (static_cast<float>(pillar.x_index) + 0.5f) *
            voxel_x;

        const float center_y =
            min_y +
            (static_cast<float>(pillar.y_index) + 0.5f) *
            voxel_y;

        // Z range is -3 ... +1 with voxel height 4 m.
        const float center_z = -1.0f;

        pfe_x_indices.push_back(pillar.x_index);
        pfe_y_indices.push_back(pillar.y_index);

        const size_t points_to_copy =
            std::min(
                pillar.points.size(),
                pfe_max_points);

        for (size_t point_index = 0;
             point_index < points_to_copy;
             ++point_index) {

          const auto &pt =
              pillar.points[point_index];

          const size_t pillar_base =
              pillar_index *
              pfe_max_points *
              pfe_features;

          auto pfe_index =
              [&](size_t feature_index) {

                if (pfe_cuda_style_layout) {

                  // Repository CUDA-style:
                  // [pillar][feature][point]
                  return
                      pillar_base +
                      feature_index *
                          pfe_max_points +
                      point_index;
                }

                // ONNX-style:
                // [pillar][point][feature]
                return
                    pillar_base +
                    point_index *
                        pfe_features +
                    feature_index;
              };

          pfe_input[pfe_index(0)] =
              pt.x;

          pfe_input[pfe_index(1)] =
              pt.y;

          pfe_input[pfe_index(2)] =
              pt.z;

          pfe_input[pfe_index(3)] =
              pt.intensity;

          pfe_input[pfe_index(4)] =
              pt.x - mean_x;

          pfe_input[pfe_index(5)] =
              pt.y - mean_y;

          pfe_input[pfe_index(6)] =
              pt.z - mean_z;

          pfe_input[pfe_index(7)] =
              pt.x - center_x;

          pfe_input[pfe_index(8)] =
              pt.y - center_y;

          pfe_input[pfe_index(9)] =
              pt.z - center_z;
        }

        ++pillar_index;
      }

      const size_t pfe_active_pillars =
          pillar_index;

      // -----------------------------------------------------
      // Run PointPillars PFE ONNX inference.
      //
      // Input:
      //   [20000, 32, 10]
      //
      // Output:
      //   [20000, 64]
      // -----------------------------------------------------

      std::vector<int64_t> pfe_input_shape = {
          20000,
          32,
          10
      };

      Ort::MemoryInfo memory_info =
          Ort::MemoryInfo::CreateCpu(
              OrtArenaAllocator,
              OrtMemTypeDefault);

      Ort::Value pfe_input_tensor =
          Ort::Value::CreateTensor<float>(
              memory_info,
              pfe_input.data(),
              pfe_input.size(),
              pfe_input_shape.data(),
              pfe_input_shape.size());

      const char *pfe_input_names[] = {
          "input.1"
      };

      const char *pfe_output_names[] = {
          "14"
      };

      auto pfe_outputs =
          pfe_session.Run(
              Ort::RunOptions{nullptr},
              pfe_input_names,
              &pfe_input_tensor,
              1,
              pfe_output_names,
              1);

      auto pfe_output_info =
          pfe_outputs[0]
              .GetTensorTypeAndShapeInfo();

      const auto pfe_output_shape =
          pfe_output_info.GetShape();

      const size_t pfe_output_count =
          pfe_output_info.GetElementCount();

      float *pfe_output_data =
          pfe_outputs[0]
              .GetTensorMutableData<float>();

      static bool pfe_inference_debug_printed = false;

      if (!pfe_inference_debug_printed) {

        std::cout
            << "\nPFE ONNX inference successful."
            << "\nPFE output shape: [";

        for (size_t i = 0;
             i < pfe_output_shape.size();
             ++i) {

          std::cout
              << pfe_output_shape[i];

          if (i + 1 <
              pfe_output_shape.size()) {

            std::cout << ",";
          }
        }

        std::cout
            << "]"
            << "\nPFE output elements: "
            << pfe_output_count
            << "\nFirst PFE output values: [";

        const size_t values_to_print =
            std::min<size_t>(
                8,
                pfe_output_count);

        for (size_t i = 0;
             i < values_to_print;
             ++i) {

          std::cout
              << pfe_output_data[i];

          if (i + 1 <
              values_to_print) {

            std::cout << ", ";
          }
        }

        std::cout << "]\n";

        pfe_inference_debug_printed = true;
      }

      // -----------------------------------------------------
      // Scatter PFE output into BEV feature map.
      //
      // PFE:
      //   [20000, 64]
      //
      // BEV / RPN input:
      //   [1, 64, 496, 432]
      // -----------------------------------------------------

      constexpr size_t bev_channels = 64;
      constexpr size_t bev_height = 496;
      constexpr size_t bev_width = 432;

      static std::vector<float> bev_input(
          bev_channels *
          bev_height *
          bev_width,
          0.0f);

      std::fill(
          bev_input.begin(),
          bev_input.end(),
          0.0f);

      for (size_t pillar = 0;
           pillar < pfe_active_pillars;
           ++pillar) {

        const int x_index =
            pfe_x_indices[pillar];

        const int y_index =
            pfe_y_indices[pillar];

        if (x_index < 0 ||
            x_index >= static_cast<int>(bev_width) ||
            y_index < 0 ||
            y_index >= static_cast<int>(bev_height)) {
          continue;
        }

        for (size_t feature = 0;
             feature < bev_channels;
             ++feature) {

          const size_t pfe_index =
              pillar * bev_channels +
              feature;

          const size_t bev_index =
              feature *
                  bev_height *
                  bev_width +
              static_cast<size_t>(y_index) *
                  bev_width +
              static_cast<size_t>(x_index);

          bev_input[bev_index] =
              pfe_output_data[pfe_index];
        }
      }

      // -----------------------------------------------------
      // Run PointPillars RPN ONNX inference.
      // -----------------------------------------------------

      std::vector<int64_t> rpn_input_shape = {
          1,
          64,
          496,
          432
      };

      Ort::Value rpn_input_tensor =
          Ort::Value::CreateTensor<float>(
              memory_info,
              bev_input.data(),
              bev_input.size(),
              rpn_input_shape.data(),
              rpn_input_shape.size());

      const char *rpn_input_names[] = {
          "input.1"
      };

      const char *rpn_output_names[] = {
          "253",
          "254",
          "256"
      };

      auto rpn_outputs =
          rpn_session.Run(
              Ort::RunOptions{nullptr},
              rpn_input_names,
              &rpn_input_tensor,
              1,
              rpn_output_names,
              3);

      static bool rpn_debug_printed = false;

      if (!rpn_debug_printed) {

        std::cout
            << "\nRPN ONNX inference successful.\n";

        for (size_t output_index = 0;
             output_index < rpn_outputs.size();
             ++output_index) {

          auto info =
              rpn_outputs[output_index]
                  .GetTensorTypeAndShapeInfo();

          auto shape =
              info.GetShape();

          const size_t count =
              info.GetElementCount();

          float *data_ptr =
              rpn_outputs[output_index]
                  .GetTensorMutableData<float>();

          std::cout
              << "RPN output "
              << output_index
              << " shape: [";

          for (size_t i = 0;
               i < shape.size();
               ++i) {

            std::cout << shape[i];

            if (i + 1 < shape.size()) {
              std::cout << ",";
            }
          }

          std::cout
              << "] elements="
              << count
              << " first values=[";

          const size_t print_count =
              std::min<size_t>(
                  6,
                  count);

          for (size_t i = 0;
               i < print_count;
               ++i) {

            std::cout
                << data_ptr[i];

            if (i + 1 < print_count) {
              std::cout << ", ";
            }
          }

          std::cout << "]\n";
        }

        rpn_debug_printed = true;
      }

      // =====================================================
      // PointPillars CPU post-processing
      // =====================================================

      struct AIDetection {
        float x;
        float y;
        float z;

        // Keep the repository's decoded x/y dimensions.
        float dx;
        float dy;
        float dz;

        float yaw;
        int class_id;
        float score;
      };

      struct AI2DPoint {
        float x;
        float y;
      };

      auto sigmoid_ai = [](float x) {
        return 1.0f / (1.0f + std::exp(-x));
      };

      auto cross_ai = [](
          const AI2DPoint &p1,
          const AI2DPoint &p2,
          const AI2DPoint &p0) {

        return
            (p1.x - p0.x) *
                (p2.y - p0.y) -
            (p2.x - p0.x) *
                (p1.y - p0.y);
      };

      auto rotate_ai = [](
          const AI2DPoint &center,
          float angle_cos,
          float angle_sin,
          AI2DPoint &p) {

        const float new_x =
            (p.x - center.x) *
                angle_cos +
            (p.y - center.y) *
                (-angle_sin) +
            center.x;

        const float new_y =
            (p.x - center.x) *
                angle_sin +
            (p.y - center.y) *
                angle_cos +
            center.y;

        p.x = new_x;
        p.y = new_y;
      };

      auto point_inside_box =
          [](
              const AIDetection &box,
              const AI2DPoint &p) {

        constexpr float margin = 1e-2f;

        const float angle_cos =
            std::cos(-box.yaw);

        const float angle_sin =
            std::sin(-box.yaw);

        const float rot_x =
            (p.x - box.x) *
                angle_cos +
            (p.y - box.y) *
                (-angle_sin);

        const float rot_y =
            (p.x - box.x) *
                angle_sin +
            (p.y - box.y) *
                angle_cos;

        return
            std::fabs(rot_x) <
                box.dx / 2.0f +
                margin &&
            std::fabs(rot_y) <
                box.dy / 2.0f +
                margin;
      };

      auto line_intersection =
          [&cross_ai](
              const AI2DPoint &p1,
              const AI2DPoint &p0,
              const AI2DPoint &q1,
              const AI2DPoint &q0,
              AI2DPoint &ans) {

        if (!(
            std::min(p0.x, p1.x) <=
                std::max(q0.x, q1.x) &&
            std::min(q0.x, q1.x) <=
                std::max(p0.x, p1.x) &&
            std::min(p0.y, p1.y) <=
                std::max(q0.y, q1.y) &&
            std::min(q0.y, q1.y) <=
                std::max(p0.y, p1.y))) {

          return false;
        }

        const float s1 =
            cross_ai(q0, p1, p0);

        const float s2 =
            cross_ai(p1, q1, p0);

        const float s3 =
            cross_ai(p0, q1, q0);

        const float s4 =
            cross_ai(q1, p1, q0);

        if (!(s1 * s2 > 0.0f &&
              s3 * s4 > 0.0f)) {

          return false;
        }

        const float s5 =
            cross_ai(q1, p1, p0);

        constexpr float eps = 1e-8f;

        if (std::fabs(s5 - s1) > eps) {

          ans.x =
              (s5 * q0.x -
               s1 * q1.x) /
              (s5 - s1);

          ans.y =
              (s5 * q0.y -
               s1 * q1.y) /
              (s5 - s1);

        } else {

          const float a0 =
              p0.y - p1.y;

          const float b0 =
              p1.x - p0.x;

          const float c0 =
              p0.x * p1.y -
              p1.x * p0.y;

          const float a1 =
              q0.y - q1.y;

          const float b1 =
              q1.x - q0.x;

          const float c1 =
              q0.x * q1.y -
              q1.x * q0.y;

          const float D =
              a0 * b1 -
              a1 * b0;

          if (std::fabs(D) < eps) {
            return false;
          }

          ans.x =
              (b0 * c1 -
               b1 * c0) /
              D;

          ans.y =
              (a1 * c0 -
               a0 * c1) /
              D;
        }

        return true;
      };

      auto rotated_overlap =
          [&rotate_ai,
           &point_inside_box,
           &line_intersection](
              const AIDetection &a,
              const AIDetection &b) {

        AI2DPoint a_corners[5];
        AI2DPoint b_corners[5];

        const float a_dx =
            a.dx / 2.0f;

        const float a_dy =
            a.dy / 2.0f;

        const float b_dx =
            b.dx / 2.0f;

        const float b_dy =
            b.dy / 2.0f;

        a_corners[0] =
            {a.x - a_dx,
             a.y - a_dy};

        a_corners[1] =
            {a.x + a_dx,
             a.y - a_dy};

        a_corners[2] =
            {a.x + a_dx,
             a.y + a_dy};

        a_corners[3] =
            {a.x - a_dx,
             a.y + a_dy};

        b_corners[0] =
            {b.x - b_dx,
             b.y - b_dy};

        b_corners[1] =
            {b.x + b_dx,
             b.y - b_dy};

        b_corners[2] =
            {b.x + b_dx,
             b.y + b_dy};

        b_corners[3] =
            {b.x - b_dx,
             b.y + b_dy};

        const AI2DPoint center_a =
            {a.x, a.y};

        const AI2DPoint center_b =
            {b.x, b.y};

        const float a_cos =
            std::cos(a.yaw);

        const float a_sin =
            std::sin(a.yaw);

        const float b_cos =
            std::cos(b.yaw);

        const float b_sin =
            std::sin(b.yaw);

        for (int k = 0; k < 4; ++k) {

          rotate_ai(
              center_a,
              a_cos,
              a_sin,
              a_corners[k]);

          rotate_ai(
              center_b,
              b_cos,
              b_sin,
              b_corners[k]);
        }

        a_corners[4] =
            a_corners[0];

        b_corners[4] =
            b_corners[0];

        AI2DPoint cross_points[24];

        AI2DPoint poly_center =
            {0.0f, 0.0f};

        int count = 0;

        for (int i = 0; i < 4; ++i) {

          for (int j = 0; j < 4; ++j) {

            AI2DPoint result;

            if (line_intersection(
                    a_corners[i + 1],
                    a_corners[i],
                    b_corners[j + 1],
                    b_corners[j],
                    result)) {

              cross_points[count++] =
                  result;

              poly_center.x +=
                  result.x;

              poly_center.y +=
                  result.y;
            }
          }
        }

        for (int k = 0; k < 4; ++k) {

          if (point_inside_box(
                  a,
                  b_corners[k])) {

            cross_points[count++] =
                b_corners[k];

            poly_center.x +=
                b_corners[k].x;

            poly_center.y +=
                b_corners[k].y;
          }

          if (point_inside_box(
                  b,
                  a_corners[k])) {

            cross_points[count++] =
                a_corners[k];

            poly_center.x +=
                a_corners[k].x;

            poly_center.y +=
                a_corners[k].y;
          }
        }

        if (count < 3) {
          return 0.0f;
        }

        poly_center.x /=
            static_cast<float>(count);

        poly_center.y /=
            static_cast<float>(count);

        std::sort(
            cross_points,
            cross_points + count,
            [&poly_center](
                const AI2DPoint &p1,
                const AI2DPoint &p2) {

              return
                  std::atan2(
                      p1.y -
                          poly_center.y,
                      p1.x -
                          poly_center.x)
                  <
                  std::atan2(
                      p2.y -
                          poly_center.y,
                      p2.x -
                          poly_center.x);
            });

        float area = 0.0f;

        for (int k = 1;
             k < count - 1;
             ++k) {

          const float ax =
              cross_points[k].x -
              cross_points[0].x;

          const float ay =
              cross_points[k].y -
              cross_points[0].y;

          const float bx =
              cross_points[k + 1].x -
              cross_points[0].x;

          const float by =
              cross_points[k + 1].y -
              cross_points[0].y;

          area +=
              ax * by -
              ay * bx;
        }

        return
            std::fabs(area) /
            2.0f;
      };

      // -----------------------------------------------------
      // RPN tensor pointers
      // -----------------------------------------------------

      const float *cls_data =
          rpn_outputs[0]
              .GetTensorData<float>();

      const float *box_data =
          rpn_outputs[1]
              .GetTensorData<float>();

      const float *dir_data =
          rpn_outputs[2]
              .GetTensorData<float>();

      constexpr int feature_x_size = 216;
      constexpr int feature_y_size = 248;

      constexpr int num_classes = 3;
      constexpr int num_anchors = 6;
      constexpr int box_values = 7;

      constexpr float score_threshold =
          0.10f;

      constexpr float nms_threshold =
          0.01f;

      constexpr float dir_offset =
          0.78539f;

      constexpr float x_min_ai =
          0.0f;

      constexpr float x_max_ai =
          69.12f;

      constexpr float y_min_ai =
          -39.68f;

      constexpr float y_max_ai =
          39.68f;

      // Anchor format:
      // dx, dy, dz, rotation
      const float anchors[6][4] = {
          {3.9f,  1.6f, 1.56f, 0.0f},
          {3.9f,  1.6f, 1.56f, 1.57f},

          {0.8f,  0.6f, 1.73f, 0.0f},
          {0.8f,  0.6f, 1.73f, 1.57f},

          {1.76f, 0.6f, 1.73f, 0.0f},
          {1.76f, 0.6f, 1.73f, 1.57f}
      };

      const float anchor_bottom[3] = {
          -1.78f,
          -0.6f,
          -0.6f
      };

      std::vector<AIDetection>
          ai_candidates;

      ai_candidates.reserve(1000);

      // -----------------------------------------------------
      // Decode every RPN location + anchor.
      // -----------------------------------------------------

      for (int row = 0;
           row < feature_y_size;
           ++row) {

        for (int col = 0;
             col < feature_x_size;
             ++col) {

          const int loc_index =
              row *
                  feature_x_size +
              col;

          const float anchor_x =
              x_min_ai +
              static_cast<float>(col) *
                  (x_max_ai -
                   x_min_ai) /
                  static_cast<float>(
                      feature_x_size -
                      1);

          const float anchor_y =
              y_min_ai +
              static_cast<float>(row) *
                  (y_max_ai -
                   y_min_ai) /
                  static_cast<float>(
                      feature_y_size -
                      1);

          for (int anchor_id = 0;
               anchor_id < num_anchors;
               ++anchor_id) {

            const int cls_offset =
                loc_index *
                    num_anchors *
                    num_classes +
                anchor_id *
                    num_classes;

            float best_score =
                sigmoid_ai(
                    cls_data[
                        cls_offset]);

            int class_id = 0;

            for (int c = 1;
                 c < num_classes;
                 ++c) {

              const float score =
                  sigmoid_ai(
                      cls_data[
                          cls_offset +
                          c]);

              if (score >
                  best_score) {

                best_score =
                    score;

                class_id =
                    c;
              }
            }

            if (best_score <
                score_threshold) {

              continue;
            }

            const int box_offset =
                loc_index *
                    num_anchors *
                    box_values +
                anchor_id *
                    box_values;

            const int dir_offset_index =
                loc_index *
                    num_anchors *
                    2 +
                anchor_id *
                    2;

            const float dxa =
                anchors[anchor_id][0];

            const float dya =
                anchors[anchor_id][1];

            const float dza =
                anchors[anchor_id][2];

            const float ra =
                anchors[anchor_id][3];

            const float za =
                dza / 2.0f +
                anchor_bottom[
                    anchor_id / 2];

            const float diagonal =
                std::sqrt(
                    dxa * dxa +
                    dya * dya);

            const float x =
                box_data[
                    box_offset] *
                    diagonal +
                anchor_x;

            const float y =
                box_data[
                    box_offset + 1] *
                    diagonal +
                anchor_y;

            const float z =
                box_data[
                    box_offset + 2] *
                    dza +
                za;

            const float dx =
                std::exp(
                    box_data[
                        box_offset + 3]) *
                dxa;

            const float dy =
                std::exp(
                    box_data[
                        box_offset + 4]) *
                dya;

            const float dz =
                std::exp(
                    box_data[
                        box_offset + 5]) *
                dza;

            const float decoded_rotation =
                box_data[
                    box_offset + 6] +
                ra;

            const int dir_label =
                dir_data[
                    dir_offset_index]
                >
                dir_data[
                    dir_offset_index +
                    1]
                ? 0
                : 1;

            const float period =
                static_cast<float>(
                    M_PI);

            const float value =
                decoded_rotation -
                dir_offset;

            const float dir_rot =
                value -
                std::floor(
                    value /
                        (period +
                         1e-8f)) *
                    period;

            const float yaw =
                dir_rot +
                dir_offset +
                period *
                    static_cast<float>(
                        dir_label);

            ai_candidates.push_back(
                {
                    x,
                    y,
                    z,
                    dx,
                    dy,
                    dz,
                    yaw,
                    class_id,
                    best_score
                });
          }
        }
      }

      // -----------------------------------------------------
      // Repository-style rotated NMS.
      // -----------------------------------------------------

      std::sort(
          ai_candidates.begin(),
          ai_candidates.end(),
          [](
              const AIDetection &a,
              const AIDetection &b) {

            return
                a.score >
                b.score;
          });

      std::vector<AIDetection>
          ai_detections;

      std::vector<bool>
          suppressed(
              ai_candidates.size(),
              false);

      for (size_t i = 0;
           i < ai_candidates.size();
           ++i) {

        if (suppressed[i]) {
          continue;
        }

        ai_detections.push_back(
            ai_candidates[i]);

        for (size_t j = i + 1;
             j < ai_candidates.size();
             ++j) {

          if (suppressed[j]) {
            continue;
          }

          const float area_a =
              ai_candidates[i].dx *
              ai_candidates[i].dy;

          const float area_b =
              ai_candidates[j].dx *
              ai_candidates[j].dy;

          const float overlap =
              rotated_overlap(
                  ai_candidates[i],
                  ai_candidates[j]);

          const float denom =
              std::max(
                  area_a +
                      area_b -
                      overlap,
                  1e-8f);

          const float iou =
              overlap /
              denom;

          if (iou >=
              nms_threshold) {

            suppressed[j] =
                true;
          }
        }
      }

      // =====================================================
      // Pedestrian hazard detection for ego braking.
      //
      // PointPillars classes:
      //   0 = Car
      //   1 = Pedestrian
      //   2 = Cyclist
      //
      // AI/KITTI frame:
      //   x = forward
      //   y = lateral
      // =====================================================

      bool pedestrian_detected_in_path = false;

      for (const auto &d : ai_detections) {

        if (d.class_id != 1) {
          continue;
        }

        if (d.score < 0.15f) {
          continue;
        }

        if (d.x > 0.0f &&
            d.x < 15.0f &&
            std::fabs(d.y) < 2.0f) {

          pedestrian_detected_in_path = true;

          std::cout
              << "\nPEDESTRIAN_IN_PATH"
              << " | Conf: " << d.score
              << " | X: " << d.x
              << " | Y: " << d.y
              << std::endl;

          break;
        }
      }

      // -----------------------------------------------------
      // Hysteresis:
      // require 2 consecutive pedestrian detections to stop,
      // and 5 clear frames before driving again.
      // -----------------------------------------------------
      static int pedestrian_hit_frames = 0;
      static int pedestrian_clear_frames = 0;

      if (pedestrian_detected_in_path) {

        ++pedestrian_hit_frames;
        pedestrian_clear_frames = 0;

        if (pedestrian_hit_frames >= 2) {
          pedestrian_hazard.store(true);
        }

      } else {

        pedestrian_hit_frames = 0;

        if (pedestrian_hazard.load()) {

          ++pedestrian_clear_frames;

          if (pedestrian_clear_frames >= 5) {

            pedestrian_hazard.store(false);
            pedestrian_clear_frames = 0;

            std::cout
                << "\nPEDESTRIAN CLEAR -> EGO MAY RESUME"
                << std::endl;
          }

        } else {
          pedestrian_clear_frames = 0;
        }
      }

      static bool ai_detection_debug_printed =
          false;

      if (!ai_detection_debug_printed) {

        const char *class_names[3] = {
            "Car",
            "Pedestrian",
            "Cyclist"
        };

        std::cout
            << "\nAI post-processing successful."
            << "\nCandidates before NMS: "
            << ai_candidates.size()
            << "\nDetections after NMS: "
            << ai_detections.size()
            << "\n";

        const size_t detections_to_print =
            std::min<size_t>(
                10,
                ai_detections.size());

        for (size_t i = 0;
             i < detections_to_print;
             ++i) {

          const auto &d =
              ai_detections[i];

          const float distance =
              std::sqrt(
                  d.x * d.x +
                  d.y * d.y);

          std::cout
              << "AI Detection "
              << i
              << " | Class: "
              << class_names[
                     d.class_id]
              << " | Confidence: "
              << d.score
              << " | X: "
              << d.x
              << " | Y: "
              << d.y
              << " | Z: "
              << d.z
              << " | DX: "
              << d.dx
              << " | DY: "
              << d.dy
              << " | DZ: "
              << d.dz
              << " | Yaw: "
              << d.yaw
              << " | Distance: "
              << distance
              << " m\n";
        }

        ai_detection_debug_printed =
            true;
      }

      std::cout
          << "AIInputPoints: "
          << ai_points.size()
          << " | AIKITTI: "
          << ai_kitti_points.size()
          << " | Pillars: "
          << pillar_map.size()
          << " | PillarPoints: "
          << pillar_point_count
          << " | PFEPillars: "
          << pfe_active_pillars
          << " | ";

      // Print one PFE tensor example only once.
      static bool pfe_debug_printed = false;

      if (!pfe_debug_printed &&
          pfe_active_pillars > 0) {

        std::cout
            << "\nPFE tensor shape: [20000,32,10]"
            << "\nFirst pillar grid: x="
            << pfe_x_indices[0]
            << " y="
            << pfe_y_indices[0]
            << "\nFirst feature vector: [";

        for (size_t f = 0;
             f < pfe_features;
             ++f) {

          std::cout << pfe_input[f];

          if (f + 1 < pfe_features) {
            std::cout << ", ";
          }
        }

        std::cout << "]\n";

        pfe_debug_printed = true;
      }

      // =====================================================
      // PCL RISK CLUSTERING
      //
      // Narrow ROI:
      // obstacle_points
      //
      // Used for:
      // tracking, distance, TTC and collision risk.
      // =====================================================

      const float cluster_distance =
          1.0f;

      const int min_cluster_size =
          5;

      const int max_cluster_size =
          5000;

      size_t cluster_count =
          0;

      float nearest_obstacle =
          std::numeric_limits<float>::max();

      std::vector<Point3D>
          cluster_centroids;

      pcl::PointCloud<pcl::PointXYZ>::Ptr
          pcl_risk_cloud(
              new pcl::PointCloud<pcl::PointXYZ>);

      pcl_risk_cloud->points.reserve(
          obstacle_points.size());

      for (const auto &p :
           obstacle_points) {

        pcl::PointXYZ pcl_point;

        pcl_point.x = p.x;
        pcl_point.y = p.y;
        pcl_point.z = p.z;

        pcl_risk_cloud->points.push_back(
            pcl_point);
      }

      pcl_risk_cloud->width =
          static_cast<uint32_t>(
              pcl_risk_cloud->points.size());

      pcl_risk_cloud->height = 1;
      pcl_risk_cloud->is_dense = true;

      const auto pcl_risk_start =
          std::chrono::steady_clock::now();

      if (!pcl_risk_cloud->empty()) {

        pcl::search::KdTree<
            pcl::PointXYZ>::Ptr
            pcl_risk_tree(
                new pcl::search::KdTree<
                    pcl::PointXYZ>);

        pcl_risk_tree->setInputCloud(
            pcl_risk_cloud);

        std::vector<pcl::PointIndices>
            risk_cluster_indices;

        pcl::EuclideanClusterExtraction<
            pcl::PointXYZ>
            pcl_risk_ec;

        pcl_risk_ec.setClusterTolerance(
            cluster_distance);

        pcl_risk_ec.setMinClusterSize(
            min_cluster_size);

        pcl_risk_ec.setMaxClusterSize(
            max_cluster_size);

        pcl_risk_ec.setSearchMethod(
            pcl_risk_tree);

        pcl_risk_ec.setInputCloud(
            pcl_risk_cloud);

        pcl_risk_ec.extract(
            risk_cluster_indices);

        cluster_count =
            risk_cluster_indices.size();

        for (const auto &indices :
             risk_cluster_indices) {

          float sum_x = 0.0f;
          float sum_y = 0.0f;
          float sum_z = 0.0f;

          for (const int idx :
               indices.indices) {

            const auto &point =
                pcl_risk_cloud->points[
                    static_cast<size_t>(idx)];

            sum_x += point.x;
            sum_y += point.y;
            sum_z += point.z;

            const float distance =
                std::sqrt(
                    point.x * point.x +
                    point.y * point.y);

            if (distance <
                nearest_obstacle) {

              nearest_obstacle =
                  distance;
            }
          }

          if (!indices.indices.empty()) {

            const float count =
                static_cast<float>(
                    indices.indices.size());

            Point3D centroid;

            centroid.x =
                sum_x / count;

            centroid.y =
                sum_y / count;

            centroid.z =
                sum_z / count;

            cluster_centroids.push_back(
                centroid);
          }
        }
      }

      const auto pcl_risk_end =
          std::chrono::steady_clock::now();

      const float pcl_risk_ms =
          std::chrono::duration<
              float,
              std::milli>(
                  pcl_risk_end -
                  pcl_risk_start)
              .count();

      std::cout
          << "PCL_RISK"
          << " | InputPoints: "
          << pcl_risk_cloud->size()
          << " | Clusters: "
          << cluster_count
          << " | Time: "
          << pcl_risk_ms
          << " ms"
          << std::endl;


      // =====================================================
      // PCL WIDE CLUSTERING
      //
      // Wide ROI:
      // clustering_points
      //
      // Used only for:
      // visualization and clustering research.
      // =====================================================

      pcl::PointCloud<pcl::PointXYZ>::Ptr
          pcl_wide_cloud(
              new pcl::PointCloud<pcl::PointXYZ>);

      pcl_wide_cloud->points.reserve(
          clustering_points.size());

      for (const auto &p :
           clustering_points) {

        pcl::PointXYZ pcl_point;

        pcl_point.x = p.x;
        pcl_point.y = p.y;
        pcl_point.z = p.z;

        pcl_wide_cloud->points.push_back(
            pcl_point);
      }

      pcl_wide_cloud->width =
          static_cast<uint32_t>(
              pcl_wide_cloud->points.size());

      pcl_wide_cloud->height = 1;
      pcl_wide_cloud->is_dense = true;

      size_t wide_cluster_count = 0;

      const auto pcl_wide_start =
          std::chrono::steady_clock::now();

      if (!pcl_wide_cloud->empty()) {

        pcl::search::KdTree<
            pcl::PointXYZ>::Ptr
            pcl_wide_tree(
                new pcl::search::KdTree<
                    pcl::PointXYZ>);

        pcl_wide_tree->setInputCloud(
            pcl_wide_cloud);

        std::vector<pcl::PointIndices>
            wide_cluster_indices;

        pcl::EuclideanClusterExtraction<
            pcl::PointXYZ>
            pcl_wide_ec;

        pcl_wide_ec.setClusterTolerance(
            cluster_distance);

        pcl_wide_ec.setMinClusterSize(
            min_cluster_size);

        pcl_wide_ec.setMaxClusterSize(
            max_cluster_size);

        pcl_wide_ec.setSearchMethod(
            pcl_wide_tree);

        pcl_wide_ec.setInputCloud(
            pcl_wide_cloud);

        pcl_wide_ec.extract(
            wide_cluster_indices);

        wide_cluster_count =
            wide_cluster_indices.size();

        cluster_vis_points.clear();
        cluster_vis_colors.clear();

        cluster_vis_points.reserve(
            pcl_wide_cloud->size());

        cluster_vis_colors.assign(
            pcl_wide_cloud->size(),
            Eigen::Vector3d(
                0.35, 0.35, 0.35));

        for (const auto &p :
             pcl_wide_cloud->points) {

          cluster_vis_points.emplace_back(
              static_cast<double>(p.x),
              static_cast<double>(-p.y),
              static_cast<double>(p.z));
        }

        for (size_t cluster_id = 0;
             cluster_id <
                 wide_cluster_indices.size();
             ++cluster_id) {

          const Eigen::Vector3d color =
              GetClusterColor(cluster_id);

          for (const int idx :
               wide_cluster_indices[
                   cluster_id].indices) {

            if (idx >= 0 &&
                static_cast<size_t>(idx) <
                    cluster_vis_colors.size()) {

              cluster_vis_colors[
                  static_cast<size_t>(idx)] =
                      color;
            }
          }
        }
      }

      const auto pcl_wide_end =
          std::chrono::steady_clock::now();

      const float pcl_wide_ms =
          std::chrono::duration<
              float,
              std::milli>(
                  pcl_wide_end -
                  pcl_wide_start)
              .count();

      std::cout
          << "PCL_WIDE"
          << " | InputPoints: "
          << pcl_wide_cloud->size()
          << " | Clusters: "
          << wide_cluster_count
          << " | Time: "
          << pcl_wide_ms
          << " ms"
          << std::endl;


      // Publish wide clustering visualization.
      {
        std::lock_guard<std::mutex> lock(
            vis_points_mutex);

        if (!cluster_vis_points.empty()) {

          latest_vis_points =
              cluster_vis_points;

          latest_vis_colors =
              cluster_vis_colors;

        } else {

          latest_vis_points =
              vis_points;

          latest_vis_colors =
              vis_colors;
        }
      }

      static bool has_previous_track = false;
      static Point3D previous_track = {0.0f, 0.0f, 0.0f};
      static int missed_frames = 0;

      const float max_match_distance = 1.5f;
      const int max_missed_frames = 3;

      bool track_valid = false;
      bool new_track_started = false;
      Point3D tracked_centroid = {0.0f, 0.0f, 0.0f};
      float tracked_distance = -1.0f;

      if (has_previous_track && !cluster_centroids.empty()) {

        float best_match_distance =
            std::numeric_limits<float>::max();

        size_t best_index = 0;

        for (size_t i = 0; i < cluster_centroids.size(); ++i) {

          float dx =
              cluster_centroids[i].x - previous_track.x;
          float dy =
              cluster_centroids[i].y - previous_track.y;

          float match_distance =
              std::sqrt(dx * dx + dy * dy);

          if (match_distance < best_match_distance) {
            best_match_distance = match_distance;
            best_index = i;
          }
        }

        if (best_match_distance < max_match_distance) {

          tracked_centroid = cluster_centroids[best_index];

          tracked_distance = std::sqrt(
              tracked_centroid.x * tracked_centroid.x +
              tracked_centroid.y * tracked_centroid.y);

          previous_track = tracked_centroid;

          track_valid = true;
          missed_frames = 0;

        } else {

          ++missed_frames;

          if (missed_frames > max_missed_frames) {
            has_previous_track = false;
            missed_frames = 0;
          }
        }

      } else if (has_previous_track) {

        ++missed_frames;

        if (missed_frames > max_missed_frames) {
          has_previous_track = false;
          missed_frames = 0;
        }
      }

      if (!has_previous_track && !cluster_centroids.empty()) {

        float closest_distance =
            std::numeric_limits<float>::max();

        size_t closest_index = 0;

        for (size_t i = 0; i < cluster_centroids.size(); ++i) {

          float d = std::sqrt(
              cluster_centroids[i].x * cluster_centroids[i].x +
              cluster_centroids[i].y * cluster_centroids[i].y);

          if (d < closest_distance) {
            closest_distance = d;
            closest_index = i;
          }
        }

        tracked_centroid = cluster_centroids[closest_index];
        tracked_distance = closest_distance;

        previous_track = tracked_centroid;

        has_previous_track = true;
        track_valid = true;
        new_track_started = true;
        missed_frames = 0;
      }

      static bool has_velocity_history = false;
      static bool has_smoothed_distance = false;
      static float previous_smoothed_distance = 0.0f;
      static float smoothed_distance = 0.0f;
      static double previous_track_time = 0.0;

      const float smoothing_alpha = 0.20f;

      float closing_speed = 0.0f;
      float ttc = -1.0f;
      const char *risk_level = "NO_TRACK";

      static int warning_counter = 0;
      static int emergency_counter = 0;
      static const char *stable_risk = "SAFE";

      double current_time = measurement->GetTimestamp();

      if (new_track_started) {
        has_velocity_history = false;
        has_smoothed_distance = false;
      }

      if (track_valid) {

        risk_level = "SAFE";

        if (!has_smoothed_distance) {
          smoothed_distance = tracked_distance;
          has_smoothed_distance = true;
        } else {
          smoothed_distance =
              smoothing_alpha * tracked_distance +
              (1.0f - smoothing_alpha) * smoothed_distance;
        }

        if (has_velocity_history) {

          double dt = current_time - previous_track_time;

          if (dt > 0.0001) {

            closing_speed =
                (previous_smoothed_distance - smoothed_distance) /
                static_cast<float>(dt);

            if (closing_speed > 0.1f) {

              ttc = smoothed_distance / closing_speed;

              if (ttc < 2.0f) {
                risk_level = "EMERGENCY";
              } else if (ttc < 4.0f) {
                risk_level = "WARNING";
              } else {
                risk_level = "SAFE";
              }
            }
          }
        }

        previous_smoothed_distance = smoothed_distance;
        previous_track_time = current_time;
        has_velocity_history = true;

      } else if (!has_previous_track) {

        has_velocity_history = false;
        has_smoothed_distance = false;
      }

      if (track_valid) {

        if (std::string(risk_level) == "EMERGENCY") {
          ++emergency_counter;
          ++warning_counter;

        } else if (std::string(risk_level) == "WARNING") {
          emergency_counter = 0;
          ++warning_counter;

        } else {
          emergency_counter = 0;
          warning_counter = 0;
        }

        if (emergency_counter >= 3) {
          stable_risk = "EMERGENCY";
        } else if (warning_counter >= 3) {
          stable_risk = "WARNING";
        } else {
          stable_risk = "SAFE";
        }

      } else {
        emergency_counter = 0;
        warning_counter = 0;
        stable_risk = "SAFE";
      }

      // End pure C++ LiDAR processing timer here.
      // Ground-truth CARLA queries below are excluded.
      const auto processing_end =
          std::chrono::steady_clock::now();

      const double processing_ms =
          std::chrono::duration<double, std::milli>(
              processing_end - processing_start).count();

      const double processing_fps =
          processing_ms > 0.0
              ? 1000.0 / processing_ms
              : 0.0;

      // =====================================================
      // OPERATIONAL AI TRACKER
      //
      // IMPORTANT:
      // This section uses ONLY PointPillars detections.
      // CARLA ground truth is NOT used for target selection,
      // tracking, TTC, or risk classification.
      // =====================================================

      static bool ai_has_previous_track = false;
      static AIDetection ai_previous_track{};

      static int ai_missed_frames = 0;
      constexpr int ai_max_missed_frames = 5;

      bool ai_track_valid = false;
      bool ai_new_track_started = false;

      AIDetection ai_tracked_detection{};

      // -----------------------------------------------------
      // 1. Try to associate current detections with the
      //    previous AI track.
      //
      // Association uses x/y center distance only.
      // -----------------------------------------------------
      if (ai_has_previous_track) {

        float best_association_distance =
            std::numeric_limits<float>::max();

        bool associated = false;

        AIDetection best_detection{};

        for (const auto &det : ai_detections) {

          // Operational forward-driving corridor.
          // No CARLA GT is used here.
          if (det.x <= 3.0f ||
              det.x >= 40.0f ||
              std::fabs(det.y) > 1.5f) {
            continue;
          }

          const float dx =
              det.x - ai_previous_track.x;

          const float dy =
              det.y - ai_previous_track.y;

          const float association_distance =
              std::sqrt(
                  dx * dx +
                  dy * dy);

          if (association_distance <
              best_association_distance) {

            best_association_distance =
                association_distance;

            best_detection =
                det;

            associated =
                true;
          }
        }

        // Allow reasonable movement between consecutive frames.
        const float ai_association_gate =
            std::min(
                3.5f,
                1.5f +
                0.5f *
                static_cast<float>(
                    ai_missed_frames));

        if (associated &&
            best_association_distance <=
                ai_association_gate) {

          ai_tracked_detection =
              best_detection;

          ai_previous_track =
              best_detection;

          ai_track_valid =
              true;

          ai_missed_frames =
              0;

        } else {

          ++ai_missed_frames;

          if (ai_missed_frames >
              ai_max_missed_frames) {

            ai_has_previous_track =
                false;

            ai_missed_frames =
                0;
          }
        }
      }

      // -----------------------------------------------------
      // 2. If no active AI track exists, start one using the
      //    nearest forward AI detection in the driving lane.
      //
      // This is operational target selection, NOT GT matching.
      // -----------------------------------------------------
      if (!ai_has_previous_track) {

        float nearest_ai_distance =
            std::numeric_limits<float>::max();

        bool candidate_found =
            false;

        AIDetection candidate{};

        for (const auto &det : ai_detections) {

          if (det.x <= 3.0f ||
              det.x >= 40.0f ||
              std::fabs(det.y) > 1.5f) {
            continue;
          }

          const float distance =
              std::sqrt(
                  det.x * det.x +
                  det.y * det.y);

          if (distance <
              nearest_ai_distance) {

            nearest_ai_distance =
                distance;

            candidate =
                det;

            candidate_found =
                true;
          }
        }

        if (candidate_found) {

          ai_tracked_detection =
              candidate;

          ai_previous_track =
              candidate;

          ai_has_previous_track =
              true;

          ai_track_valid =
              true;

          ai_new_track_started =
              true;

          ai_missed_frames =
              0;
        }
      }

      // -----------------------------------------------------
      // 3. AI distance smoothing + closing speed + TTC.
      // -----------------------------------------------------
      static bool ai_has_smoothed_distance =
          false;

      static bool ai_has_velocity_history =
          false;

      static float ai_smoothed_distance =
          0.0f;

      static float ai_previous_smoothed_distance =
          0.0f;

      static double ai_previous_track_time =
          0.0;

      // Last 5 valid AI measurements for robust
      // closing-speed and TTC estimation.
      static float ai_distance_history[5] = {
          0.0f, 0.0f, 0.0f, 0.0f, 0.0f
      };

      static double ai_time_history[5] = {
          0.0, 0.0, 0.0, 0.0, 0.0
      };

      static int ai_history_count = 0;

      constexpr float ai_smoothing_alpha =
          0.20f;

      float ai_track_distance =
          -1.0f;

      float ai_closing_speed =
          0.0f;

      float ai_ttc =
          -1.0f;

      const char *ai_raw_risk =
          "NO_TRACK";

      static int ai_warning_counter =
          0;

      static int ai_emergency_counter =
          0;

      static std::string ai_stable_risk =
          "SAFE";

      const double ai_current_time =
          measurement->GetTimestamp();

      if (ai_new_track_started) {

        ai_has_smoothed_distance =
            false;

        ai_has_velocity_history =
            false;

        ai_history_count =
            0;
      }

      if (ai_track_valid) {

        ai_track_distance =
            std::sqrt(
                ai_tracked_detection.x *
                    ai_tracked_detection.x +
                ai_tracked_detection.y *
                    ai_tracked_detection.y);

        if (!ai_has_smoothed_distance) {

          ai_smoothed_distance =
              ai_track_distance;

          ai_has_smoothed_distance =
              true;

        } else {

          ai_smoothed_distance =
              ai_smoothing_alpha *
                  ai_track_distance +
              (1.0f -
               ai_smoothing_alpha) *
                  ai_smoothed_distance;
        }

        ai_raw_risk =
            "SAFE";

        // -------------------------------------------------
        // Add current valid AI measurement to a 5-sample
        // history window.
        // -------------------------------------------------
        if (ai_history_count < 5) {

          ai_distance_history[ai_history_count] =
              ai_smoothed_distance;

          ai_time_history[ai_history_count] =
              ai_current_time;

          ++ai_history_count;

        } else {

          // Shift older measurements left.
          for (int i = 0; i < 4; ++i) {

            ai_distance_history[i] =
                ai_distance_history[i + 1];

            ai_time_history[i] =
                ai_time_history[i + 1];
          }

          ai_distance_history[4] =
              ai_smoothed_distance;

          ai_time_history[4] =
              ai_current_time;
        }

        // -------------------------------------------------
        // Estimate distance trend using linear regression.
        //
        // Distance decreasing with time gives a negative
        // slope, so closing speed = -slope.
        // At least 3 valid measurements are required.
        // -------------------------------------------------
        if (ai_history_count >= 3) {

          double mean_time = 0.0;
          double mean_distance = 0.0;

          for (int i = 0;
               i < ai_history_count;
               ++i) {

            mean_time +=
                ai_time_history[i];

            mean_distance +=
                ai_distance_history[i];
          }

          mean_time /=
              static_cast<double>(
                  ai_history_count);

          mean_distance /=
              static_cast<double>(
                  ai_history_count);

          double numerator = 0.0;
          double denominator = 0.0;

          for (int i = 0;
               i < ai_history_count;
               ++i) {

            const double dt =
                ai_time_history[i] -
                mean_time;

            const double dd =
                static_cast<double>(
                    ai_distance_history[i]) -
                mean_distance;

            numerator +=
                dt * dd;

            denominator +=
                dt * dt;
          }

          if (denominator > 1e-8) {

            const double distance_slope =
                numerator /
                denominator;

            ai_closing_speed =
                static_cast<float>(
                    -distance_slope);

            if (ai_closing_speed > 0.1f) {

              ai_ttc =
                  ai_smoothed_distance /
                  ai_closing_speed;

              if (ai_ttc < 2.0f) {

                ai_raw_risk =
                    "EMERGENCY";

              } else if (ai_ttc < 4.0f) {

                ai_raw_risk =
                    "WARNING";

              } else {

                ai_raw_risk =
                    "SAFE";
              }
            }
          }
        }

        ai_previous_smoothed_distance =
            ai_smoothed_distance;

        ai_previous_track_time =
            ai_current_time;

        ai_has_velocity_history =
            true;

        // -----------------------------------------------
        // Stable AI risk:
        // 3 consecutive danger frames required.
        // -----------------------------------------------
        if (std::string(ai_raw_risk) ==
            "EMERGENCY") {

          ++ai_emergency_counter;
          ++ai_warning_counter;

        } else if (std::string(ai_raw_risk) ==
                   "WARNING") {

          ai_emergency_counter =
              0;

          ++ai_warning_counter;

        } else {

          ai_emergency_counter =
              0;

          ai_warning_counter =
              0;
        }

        if (ai_emergency_counter >= 3) {

          ai_stable_risk =
              "EMERGENCY";

        } else if (ai_warning_counter >= 3) {

          ai_stable_risk =
              "WARNING";

        } else if (std::string(ai_raw_risk) ==
                   "SAFE") {

          ai_stable_risk =
              "SAFE";
        }

      } else {

        // Keep the track alive for short detection dropouts,
        // but do not invent a current measurement.
        ai_raw_risk =
            "NO_TRACK";

        if (!ai_has_previous_track) {

          ai_has_smoothed_distance =
              false;

          ai_has_velocity_history =
              false;

          ai_warning_counter =
              0;

          ai_emergency_counter =
              0;

          ai_stable_risk =
              "NO_TRACK";
        }
      }

      // -----------------------------------------------------
      // AI operational tracker output.
      // -----------------------------------------------------
      const char *ai_track_class_names[3] = {
          "Car",
          "Pedestrian",
          "Cyclist"
      };

      std::cout
          << "\nAI_TRACK"
          << " | Frame: "
          << measurement->GetFrame()
          << " | Valid: "
          << (ai_track_valid ? "YES" : "NO");

      if (ai_track_valid) {

        std::cout
            << " | Class: "
            << ai_track_class_names[
                   ai_tracked_detection.class_id]
            << " | Conf: "
            << ai_tracked_detection.score
            << " | X: "
            << ai_tracked_detection.x
            << " | Y: "
            << ai_tracked_detection.y
            << " | Dist: "
            << ai_track_distance
            << " m"
            << " | SmoothDist: "
            << ai_smoothed_distance
            << " m"
            << " | ClosingSpeed: "
            << ai_closing_speed
            << " m/s";

        if (ai_ttc > 0.0f) {

          std::cout
              << " | TTC: "
              << ai_ttc
              << " s";

        } else {

          std::cout
              << " | TTC: N/A";
        }

        std::cout
            << " | RawRisk: "
            << ai_raw_risk
            << " | StableRisk: "
            << ai_stable_risk;
      }

      std::cout
          << std::endl;

      // -----------------------------------------------------
      // CARLA ground truth for the stationary target vehicle.
      // Convert target world position into ego coordinates.
      // -----------------------------------------------------
      // Use the transforms captured at the beginning
      // of this LiDAR callback so GT corresponds much more
      // closely to the sensor frame being processed.
      const auto ego_tf =
          lidar_frame_ego_tf;

      const auto target_tf =
          lidar_frame_target_tf;

      const float dx =
          target_tf.location.x - ego_tf.location.x;

      const float dy =
          target_tf.location.y - ego_tf.location.y;

      const float yaw_rad =
          ego_tf.rotation.yaw *
          static_cast<float>(M_PI / 180.0);

      const float gt_x =
          std::cos(yaw_rad) * dx +
          std::sin(yaw_rad) * dy;

      const float gt_y =
          -std::sin(yaw_rad) * dx +
           std::cos(yaw_rad) * dy;

      const float gt_distance =
          std::sqrt(gt_x * gt_x + gt_y * gt_y);

      // -----------------------------------------------------
      // Ground-truth matching for AI EVALUATION.
      //
      // Important:
      // PointPillars inference is already complete here.
      // CARLA ground truth is used only to evaluate which
      // prediction corresponds to the known target vehicle.
      //
      // CARLA ego coordinates:
      //   x = forward, y = right
      //
      // AI/KITTI coordinates:
      //   x = forward, y = left
      //
      // Therefore:
      //   AI-frame GT x =  gt_x
      //   AI-frame GT y = -gt_y
      // -----------------------------------------------------

      const float ai_gt_x =
          gt_x;

      const float ai_gt_y =
          -gt_y;

      // -----------------------------------------------------
      // Diagnostic: inspect CARLA LiDAR points around the
      // known target vehicle in the AI/KITTI coordinate frame.
      //
      // We deliberately inspect RAW CARLA LiDAR points here
      // before the PointPillars z-range filter.
      // -----------------------------------------------------
      size_t target_near_points = 0;
      size_t target_kitti_valid_points = 0;

      float target_min_x =
          std::numeric_limits<float>::max();
      float target_max_x =
          std::numeric_limits<float>::lowest();

      float target_min_y =
          std::numeric_limits<float>::max();
      float target_max_y =
          std::numeric_limits<float>::lowest();

      float target_min_z =
          std::numeric_limits<float>::max();
      float target_max_z =
          std::numeric_limits<float>::lowest();

      float target_min_intensity =
          std::numeric_limits<float>::max();
      float target_max_intensity =
          std::numeric_limits<float>::lowest();

      double target_intensity_sum = 0.0;

      for (const auto &p : ai_points) {

        const float px = p.x;
        const float py = -p.y;
        const float pz = p.z;

        // Tesla-centred diagnostic window.
        //
        // X +/- 3 m covers approximately the vehicle length.
        // Y +/- 1.5 m covers approximately the vehicle width.
        if (std::fabs(px - ai_gt_x) <= 3.0f &&
            std::fabs(py - ai_gt_y) <= 1.5f) {

          ++target_near_points;

          target_min_x = std::min(target_min_x, px);
          target_max_x = std::max(target_max_x, px);

          target_min_y = std::min(target_min_y, py);
          target_max_y = std::max(target_max_y, py);

          target_min_z = std::min(target_min_z, pz);
          target_max_z = std::max(target_max_z, pz);

          target_min_intensity =
              std::min(target_min_intensity, p.intensity);

          target_max_intensity =
              std::max(target_max_intensity, p.intensity);

          target_intensity_sum += p.intensity;

          if (px >= 0.0f && px <= 69.12f &&
              py >= -39.68f && py <= 39.68f &&
              pz >= -3.0f && pz <= 1.0f) {

            ++target_kitti_valid_points;
          }
        }
      }

      if (target_near_points > 0) {

        const double target_mean_intensity =
            target_intensity_sum /
            static_cast<double>(target_near_points);

        std::cout
            << "TARGET_LIDAR"
            << " | GTDist: " << gt_distance
            << " | NearPoints: " << target_near_points
            << " | KITTIValid: " << target_kitti_valid_points
            << " | X:[" << target_min_x
            << "," << target_max_x << "]"
            << " | Y:[" << target_min_y
            << "," << target_max_y << "]"
            << " | Z:[" << target_min_z
            << "," << target_max_z << "]"
            << " | Intensity:["
            << target_min_intensity
            << "," << target_max_intensity << "]"
            << " | MeanIntensity: "
            << target_mean_intensity
            << std::endl;
      }
      else {
        std::cout
            << "TARGET_LIDAR"
            << " | GTDist: " << gt_distance
            << " | NearPoints: 0"
            << std::endl;
      }

      constexpr float ai_match_threshold =
          2.0f;

      bool ai_target_valid =
          false;

      AIDetection ai_target{};

      float ai_center_error =
          std::numeric_limits<float>::max();

      // Keep the closest prediction even if it later
      // fails the 2 m matching threshold.
      bool ai_closest_found =
          false;

      AIDetection ai_closest_detection{};

      float ai_closest_center_error =
          std::numeric_limits<float>::max();

      for (const auto &det :
           ai_detections) {

        const float error_x =
            det.x -
            ai_gt_x;

        const float error_y =
            det.y -
            ai_gt_y;

        const float center_error =
            std::sqrt(
                error_x * error_x +
                error_y * error_y);

        if (center_error <
            ai_closest_center_error) {

          ai_closest_center_error =
              center_error;

          ai_closest_detection =
              det;

          ai_closest_found =
              true;
        }
      }

      // -----------------------------------------------------
      // Diagnostic: show AI predictions close to the true
      // CARLA target position.
      //
      // This is evaluation/debug information only.
      // It does not affect inference or target matching.
      // -----------------------------------------------------
      if (gt_distance <= 17.0f) {

        const char *debug_class_names[3] = {
            "Car",
            "Pedestrian",
            "Cyclist"
        };

        std::cout
            << "\nAI_NEAR_GT"
            << " | Frame: "
            << measurement->GetFrame()
            << " | GT_X: "
            << ai_gt_x
            << " | GT_Y: "
            << ai_gt_y
            << " | GTDist: "
            << gt_distance
            << " m";

        int nearby_count = 0;

        for (const auto &det :
             ai_detections) {

          const float ex =
              det.x - ai_gt_x;

          const float ey =
              det.y - ai_gt_y;

          const float center_error =
              std::sqrt(
                  ex * ex +
                  ey * ey);

          if (center_error <= 6.0f) {

            const float det_distance =
                std::sqrt(
                    det.x * det.x +
                    det.y * det.y);

            std::cout
                << "\n  NearbyAI"
                << " | Class: "
                << debug_class_names[
                       det.class_id]
                << " | Conf: "
                << det.score
                << " | X: "
                << det.x
                << " | Y: "
                << det.y
                << " | Dist: "
                << det_distance
                << " m"
                << " | CenterError: "
                << center_error
                << " m";

            ++nearby_count;
          }
        }

        if (nearby_count == 0) {
          std::cout
              << "\n  NearbyAI: NONE within 6 m";
        }

        std::cout << std::endl;
      }

      float ai_target_distance =
          -1.0f;

      float ai_distance_error =
          -1.0f;

      bool ai_class_correct =
          false;

      if (ai_closest_found &&
          ai_closest_center_error <=
              ai_match_threshold) {

        ai_target_valid =
            true;

        ai_target =
            ai_closest_detection;

        ai_center_error =
            ai_closest_center_error;

        ai_target_distance =
            std::sqrt(
                ai_target.x *
                    ai_target.x +
                ai_target.y *
                    ai_target.y);

        ai_distance_error =
            ai_target_distance -
            gt_distance;

        // Controlled target is now the crossing pedestrian.
        // PointPillars:
        // 0 = Car, 1 = Pedestrian, 2 = Cyclist.
        ai_class_correct =
            (ai_target.class_id == 1);
      }

      float distance_error = -1.0f;

      if (track_valid) {
        distance_error =
            tracked_distance - gt_distance;
      }

      // -----------------------------------------------------
      // Ground truth to the target vehicle's rear surface.
      // For this controlled scenario the target is aligned
      // with the ego vehicle on a flat road.
      // -----------------------------------------------------
      const auto target_bbox =
          crossing_walker->GetBoundingBox();

      const float target_yaw_rad =
          target_tf.rotation.yaw *
          static_cast<float>(M_PI / 180.0);

      // Rear-most point of bounding box in target coordinates.
      const float rear_local_x =
          target_bbox.location.x - target_bbox.extent.x;

      const float rear_local_y =
          target_bbox.location.y;

      // Convert rear point from target coordinates to world.
      const float rear_world_x =
          target_tf.location.x +
          std::cos(target_yaw_rad) * rear_local_x -
          std::sin(target_yaw_rad) * rear_local_y;

      const float rear_world_y =
          target_tf.location.y +
          std::sin(target_yaw_rad) * rear_local_x +
          std::cos(target_yaw_rad) * rear_local_y;

      // World -> ego coordinates.
      const float rear_dx =
          rear_world_x - ego_tf.location.x;

      const float rear_dy =
          rear_world_y - ego_tf.location.y;

      const float gt_surface_x =
          std::cos(yaw_rad) * rear_dx +
          std::sin(yaw_rad) * rear_dy;

      const float gt_surface_y =
          -std::sin(yaw_rad) * rear_dx +
           std::cos(yaw_rad) * rear_dy;

      const float gt_surface_distance =
          std::sqrt(
              gt_surface_x * gt_surface_x +
              gt_surface_y * gt_surface_y);

      float surface_error = -1.0f;

      if (track_valid) {
        surface_error =
            tracked_distance - gt_surface_distance;
      }

      std::cout
          << "Frame " << measurement->GetFrame()
          << " | Raw: " << measurement->size()
          << " | ROI: " << roi_points
          << " | Non-ground: " << non_ground_points
          << " | Clusters: " << cluster_count;

      if (nearest_obstacle < std::numeric_limits<float>::max()) {
        std::cout << " | Nearest: " << nearest_obstacle << " m";
      }

      if (track_valid) {
        std::cout
            << " | TrackX: " << tracked_centroid.x
            << " | TrackY: " << tracked_centroid.y
            << " | TrackDist: " << tracked_distance << " m"
            << " | SmoothDist: " << smoothed_distance << " m"
            << " | ClosingSpeed: " << closing_speed << " m/s";

        if (ttc > 0.0f) {
          std::cout << " | TTC: " << ttc << " s";
        } else {
          std::cout << " | TTC: N/A";
        }

        std::cout
            << " | RawRisk: " << risk_level
            << " | StableRisk: " << stable_risk;
      }

      std::cout
          << " | GTX: " << gt_x
          << " | GTY: " << gt_y
          << " | GTDist: " << gt_distance << " m"
          << " | GTSurfaceDist: "
          << gt_surface_distance << " m";

      const char *ai_class_names[3] = {
          "Car",
          "Pedestrian",
          "Cyclist"
      };

      if (ai_target_valid) {

        std::cout
            << " | AITargetMatch: YES"
            << " | AIClass: "
            << ai_class_names[
                   ai_target.class_id]
            << " | AIConf: "
            << ai_target.score
            << " | AIX: "
            << ai_target.x
            << " | AIY: "
            << ai_target.y
            << " | AIDist: "
            << ai_target_distance
            << " m"
            << " | AICenterError: "
            << ai_center_error
            << " m"
            << " | AIDistError: "
            << ai_distance_error
            << " m"
            << " | AIClassCorrect: "
            << (ai_class_correct
                    ? "YES"
                    : "NO");

      } else {

        std::cout
            << " | AITargetMatch: NO";

        if (ai_closest_found) {

          std::cout
              << " | AIBestCenterError: "
              << ai_closest_center_error
              << " m";
        }
      }

      if (track_valid) {
        std::cout
            << " | DistError: "
            << distance_error << " m"
            << " | SurfaceError: "
            << surface_error << " m";
      }

      std::cout << std::endl;

      csv << measurement->GetFrame() << ","
          << measurement->size() << ","
          << roi_points << ","
          << non_ground_points << ","
          << cluster_count << ",";

      if (nearest_obstacle < std::numeric_limits<float>::max()) {
        csv << nearest_obstacle;
      } else {
        csv << -1.0f;
      }

      csv << ",";

      if (track_valid) {
        csv << tracked_centroid.x << ","
            << tracked_centroid.y << ","
            << tracked_distance << ","
            << smoothed_distance << ","
            << closing_speed << ","
            << ttc << ","
            << risk_level << ","
            << stable_risk;
      } else {
        csv << "-1,-1,-1,-1,0,-1,NO_TRACK,SAFE";
      }

      std::cout
          << " | ProcessingTime: " << processing_ms << " ms"
          << " | ProcessingFPS: " << processing_fps
          << std::endl;

      csv << ","
          << gt_x << ","
          << gt_y << ","
          << gt_distance << ","
          << distance_error << ","
          << gt_surface_x << ","
          << gt_surface_y << ","
          << gt_surface_distance << ","
          << surface_error << ","
          << processing_ms << ","
          << processing_fps << ",";

      if (ai_track_valid) {

        const char *csv_ai_class_names[3] = {
            "Car",
            "Pedestrian",
            "Cyclist"
        };

        csv << 1 << ","
            << csv_ai_class_names[
                   ai_tracked_detection.class_id] << ","
            << ai_tracked_detection.score << ","
            << ai_tracked_detection.x << ","
            << ai_tracked_detection.y << ","
            << ai_track_distance << ","
            << ai_smoothed_distance << ","
            << ai_closing_speed << ","
            << ai_ttc << ","
            << ai_raw_risk << ","
            << ai_stable_risk << ","
            << ai_center_error << ","
            << ai_distance_error;

      } else {

        csv << "0,NO_TRACK,0,-1,-1,-1,-1,0,-1,NO_TRACK,"
            << ai_stable_risk
            << ",-1,-1";
      }

      // -----------------------------------------------------
      // Pedestrian experiment control state.
      // -----------------------------------------------------
      const bool csv_pedestrian_hazard =
          pedestrian_hazard.load();

      const bool csv_obstacle_hazard =
          obstacle_hazard.load();

      std::string control_state;

      if (csv_pedestrian_hazard ||
          csv_obstacle_hazard) {

        if (ego_speed < 0.15f) {
          control_state = "STOPPED";
        } else {
          control_state = "BRAKING";
        }

      } else {

        if (ego_speed < 0.50f) {
          control_state = "RESUMING";
        } else {
          control_state = "DRIVING";
        }
      }

      csv << ","
          << ego_speed << ","
          << (csv_pedestrian_hazard ? 1 : 0) << ","
          << (csv_obstacle_hazard ? 1 : 0) << ","
          << control_state;

      csv << "\n";

      // Signal that this LiDAR frame has completely
      // finished processing.
      {
        std::lock_guard<std::mutex> lock(
            lidar_processing_mutex);

        last_processed_lidar_frame =
            measurement->GetFrame();
      }

      lidar_processing_cv.notify_all();
    });

    // ---------------------------------------------------------
    // Synchronous controlled approach.
    //
    // One iteration:
    //   1. Apply control
    //   2. Advance CARLA by exactly 0.1 s
    //   3. Wait until that LiDAR frame + AI processing finishes
    //   4. Check target separation
    // ---------------------------------------------------------
    std::cout
        << "\nPedestrian stop/resume experiment started."
        << "\nPress Ctrl+C when you want to stop the simulation."
        << std::endl;

    bool previous_braking_state = false;

    while (true) {

      const bool pedestrian_stop =
          pedestrian_hazard.load();

      const bool obstacle_stop =
          obstacle_hazard.load();

      const bool should_brake =
          pedestrian_stop ||
          obstacle_stop;

      carla::rpc::VehicleControl dynamic_control;

      // =====================================================
      // CARLA waypoint-based lane following.
      // The LiDAR system remains responsible for obstacle
      // detection and braking.
      // =====================================================

      float lane_steer = 0.0f;

      const auto lane_ego_tf =
          vehicle->GetTransform();

      auto current_waypoint =
          map->GetWaypoint(
              lane_ego_tf.location);

      if (current_waypoint != nullptr) {

        // Look approximately 8 m ahead along the current lane.
        auto future_waypoints =
            current_waypoint->GetNext(
                8.0);

        if (!future_waypoints.empty()) {

          const auto lane_target_tf =
              future_waypoints.front()
                  ->GetTransform();

          const float dx =
              lane_target_tf.location.x -
              lane_ego_tf.location.x;

          const float dy =
              lane_target_tf.location.y -
              lane_ego_tf.location.y;

          // Desired heading toward the lane waypoint.
          const float desired_yaw =
              std::atan2(dy, dx) *
              180.0f /
              static_cast<float>(M_PI);

          float yaw_error =
              desired_yaw -
              lane_ego_tf.rotation.yaw;

          // Normalize to [-180, 180].
          while (yaw_error > 180.0f) {
            yaw_error -= 360.0f;
          }

          while (yaw_error < -180.0f) {
            yaw_error += 360.0f;
          }

          // Simple proportional steering controller.
          lane_steer =
              0.04f *
              yaw_error;

          // Prevent excessive steering.
          if (lane_steer > 0.45f) {
            lane_steer = 0.45f;
          }

          if (lane_steer < -0.45f) {
            lane_steer = -0.45f;
          }
        }
      }

      dynamic_control.steer =
          lane_steer;
      dynamic_control.hand_brake = false;
      dynamic_control.reverse = false;
      dynamic_control.manual_gear_shift = false;

      if (should_brake) {

        dynamic_control.throttle = 0.0f;
        dynamic_control.brake = 1.0f;

      } else {

        dynamic_control.throttle = 0.25f;
        dynamic_control.brake = 0.0f;
      }

      if (should_brake !=
          previous_braking_state) {

        if (should_brake) {

          std::cout
              << "\n>>> PEDESTRIAN HAZARD - EGO BRAKING <<<"
              << std::endl;

        } else {

          std::cout
              << "\n>>> PATH CLEAR - EGO RESUMING <<<"
              << std::endl;
        }

        previous_braking_state =
            should_brake;
      }

      vehicle->ApplyControl(
          dynamic_control);

      const uint64_t simulation_frame =
          world.Tick(2s);

      {
        std::unique_lock<std::mutex> lock(
            lidar_processing_mutex);

        const bool processed =
            lidar_processing_cv.wait_for(
                lock,
                std::chrono::seconds(5),
                [&]() {
                  return
                      last_processed_lidar_frame >=
                      simulation_frame;
                });

        if (!processed) {
          std::cout
              << "Warning: timed out waiting for LiDAR frame "
              << simulation_frame
              << std::endl;
        }
      }

      // -----------------------------------------------------
      // Stop pedestrian after crossing to opposite side.
      // -----------------------------------------------------
      static bool pedestrian_crossing_finished =
          false;

      if (!pedestrian_crossing_finished) {

        const auto walker_now =
            crossing_walker->GetTransform().location;

        const float walker_dx =
            walker_now.x -
            crossing_tf.location.x;

        const float walker_dy =
            walker_now.y -
            crossing_tf.location.y;

        const float walker_lateral =
            walker_dx *
                crossing_right.x +
            walker_dy *
                crossing_right.y;

        // Started at approximately +4 m.
        // Crossing is complete near -4 m.
        if (walker_lateral <= -4.0f) {

          carla::rpc::WalkerControl
              stop_walker;

          stop_walker.direction =
              cg::Vector3D{
                  0.0f,
                  0.0f,
                  0.0f};

          stop_walker.speed =
              0.0f;

          stop_walker.jump =
              false;

          crossing_walker->ApplyControl(
              stop_walker);

          pedestrian_crossing_finished =
              true;

          std::cout
              << "\n>>> PEDESTRIAN FINISHED CROSSING <<<"
              << std::endl;
        }
      }

      // -----------------------------------------------------
      // Start pedestrian crossing only when ego is close enough.
      // -----------------------------------------------------
      static bool pedestrian_crossing_started =
          false;

      if (!pedestrian_crossing_started) {

        const auto ego_position =
            vehicle->GetTransform().location;

        const float crossing_dx =
            crossing_tf.location.x -
            ego_position.x;

        const float crossing_dy =
            crossing_tf.location.y -
            ego_position.y;

        const float distance_to_crossing =
            std::sqrt(
                crossing_dx * crossing_dx +
                crossing_dy * crossing_dy);

        if (distance_to_crossing <= 12.0f) {

          carla::rpc::WalkerControl
              start_crossing;

          start_crossing.direction =
              cg::Vector3D{
                  -crossing_right.x,
                  -crossing_right.y,
                  0.0f};

          start_crossing.speed =
              1.0f;

          start_crossing.jump =
              false;

          crossing_walker->ApplyControl(
              start_crossing);

          pedestrian_crossing_started =
              true;

          std::cout
              << "\n>>> PEDESTRIAN STARTED CROSSING"
              << " | Ego distance: "
              << distance_to_crossing
              << " m <<<"
              << std::endl;
        }
      }

      const auto ego_now =
          vehicle->GetTransform().location;

      const auto target_now =
          crossing_walker->GetTransform().location;

      const float dx =
          target_now.x -
          ego_now.x;

      const float dy =
          target_now.y -
          ego_now.y;

      const float gt_separation =
          std::sqrt(
              dx * dx +
              dy * dy);

      // -----------------------------------------------------
      // Third-person spectator camera following the ego car.
      // About 7 m behind and 3.5 m above the vehicle.
      // Updated every synchronous simulation frame.
      // -----------------------------------------------------
      const auto ego_camera_tf =
          vehicle->GetTransform();

      const auto ego_forward =
          ego_camera_tf.GetForwardVector();

      auto follow_camera_tf =
          ego_camera_tf;

      follow_camera_tf.location.x -=
          7.0f * ego_forward.x;

      follow_camera_tf.location.y -=
          7.0f * ego_forward.y;

      follow_camera_tf.location.z +=
          3.5f;

      // Keep camera aligned with the ego direction
      // while looking slightly downward.
      follow_camera_tf.rotation.pitch = -15.0f;

      spectator->SetTransform(
          follow_camera_tf);

      std::cout
          << "SYNC"
          << " | Frame: "
          << simulation_frame
          << " | EgoX: "
          << ego_now.x
          << " | EgoY: "
          << ego_now.y
          << " | TargetX: "
          << target_now.x
          << " | TargetY: "
          << target_now.y
          << " | Separation: "
          << gt_separation
          << " m"
          << std::endl;

    }

    // Apply full brake after the controlled approach.
    carla::rpc::VehicleControl stop_control;
    stop_control.throttle = 0.0f;
    stop_control.steer = 0.0f;
    stop_control.brake = 1.0f;
    stop_control.hand_brake = false;
    stop_control.reverse = false;
    stop_control.manual_gear_shift = false;

    vehicle->ApplyControl(stop_control);

    std::this_thread::sleep_for(
        std::chrono::milliseconds(500));

    vis_running = false;

    if (vis_thread.joinable()) {
      vis_thread.join();
    }

    lidar->Stop();
    lidar->Destroy();
    csv.close();

/*
    // Find a camera blueprint.
    auto camera_bp = blueprint_library->Find("sensor.camera.semantic_segmentation");
    EXPECT_TRUE(camera_bp != nullptr);

    // Spawn a camera attached to the vehicle.
    auto camera_transform = cg::Transform{
        cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
        cg::Rotation{-15.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
    auto cam_actor = world.SpawnActor(*camera_bp, camera_transform, actor.get());
    auto camera = boost::static_pointer_cast<cc::Sensor>(cam_actor);

    // Register a callback to save images to disk.
    camera->Listen([](auto data) {
        auto image = boost::static_pointer_cast<csd::Image>(data);
        EXPECT_TRUE(image != nullptr);
        SaveSemSegImageToDisk(*image);
    });

    std::this_thread::sleep_for(10s);

    // Remove actors from the simulation.
    camera->Destroy();
*/
    vehicle->SetAutopilot(false);

    // Restore CARLA settings before leaving the experiment.
    world.ApplySettings(
        original_world_settings,
        2s);

    std::this_thread::sleep_for(200ms);

    // Stop and destroy pedestrian AI controllers first.
    for (auto &controller :
         walker_controllers) {

      if (controller != nullptr) {
        controller->Stop();
        controller->Destroy();
      }
    }

    // Destroy mixed-scenario actors first.
    for (auto &scenario_actor :
         scenario_actors) {

      if (scenario_actor != nullptr) {
        scenario_actor->Destroy();
      }
    }

    target_vehicle->Destroy();
    vehicle->Destroy();

    std::cout << "Actors destroyed." << std::endl;

  } catch (const cc::TimeoutException &e) {
    std::cout << '\n' << e.what() << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cout << "\nException: " << e.what() << std::endl;
    return 2;
  }
}
