#include <cmath>
#include <mutex>
#include <string>

#include <ignition/gazebo/Model.hh>
#include <ignition/gazebo/Link.hh>
#include <ignition/gazebo/System.hh>
#include <ignition/gazebo/components/Pose.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>
#include <ignition/msgs/twist.pb.h>
#include <ignition/plugin/Register.hh>
#include <ignition/transport/Node.hh>

namespace roboticsstudio
{
class PlanarVelocityControl final:
    public ignition::gazebo::System,
    public ignition::gazebo::ISystemConfigure,
    public ignition::gazebo::ISystemPreUpdate
{
  public: void Configure(
      const ignition::gazebo::Entity &_entity,
      const std::shared_ptr<const sdf::Element> &_sdf,
      ignition::gazebo::EntityComponentManager &_ecm,
      ignition::gazebo::EventManager &) override
  {
    this->model = ignition::gazebo::Model(_entity);
    this->link = ignition::gazebo::Link(
        this->model.CanonicalLink(_ecm));
    this->link.EnableVelocityChecks(_ecm);
    if (_sdf && _sdf->HasElement("topic"))
      this->topic = _sdf->Get<std::string>("topic");
    if (this->topic.empty())
      this->topic = "/model/" + this->model.Name(_ecm) + "/cmd_vel";

    this->node.Subscribe(
        this->topic, &PlanarVelocityControl::OnCommand, this);
  }

  public: void PreUpdate(
      const ignition::gazebo::UpdateInfo &,
      ignition::gazebo::EntityComponentManager &_ecm) override
  {
    auto pose = _ecm.Component<ignition::gazebo::components::Pose>(
        this->model.Entity());
    if (!pose || !this->link.Valid(_ecm))
      return;

    ignition::msgs::Twist command;
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      command = this->command;
    }

    auto velocity = this->link.WorldLinearVelocity(_ecm).value_or(
        ignition::math::Vector3d::Zero);
    const auto commanded = pose->Data().Rot().RotateVector(
        ignition::math::Vector3d(command.linear().x(), command.linear().y(), 0.0));
    velocity.X(commanded.X());
    velocity.Y(commanded.Y());
    const auto bodyVelocity = pose->Data().Rot().Inverse().RotateVector(
        velocity);
    // Preserve world Z so gravity and terrain contacts control vertical motion.
    this->link.SetLinearVelocity(_ecm, bodyVelocity);
    this->link.SetAngularVelocity(
        _ecm,
        ignition::math::Vector3d(0.0, 0.0, command.angular().z()));
  }

  private: void OnCommand(const ignition::msgs::Twist &_message)
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->command = _message;
  }

  private: ignition::gazebo::Model model;
  private: ignition::gazebo::Link link;
  private: ignition::transport::Node node;
  private: std::string topic;
  private: ignition::msgs::Twist command;
  private: std::mutex mutex;
};
}

IGNITION_ADD_PLUGIN(
    roboticsstudio::PlanarVelocityControl,
    ignition::gazebo::System,
    ignition::gazebo::ISystemConfigure,
    ignition::gazebo::ISystemPreUpdate);
