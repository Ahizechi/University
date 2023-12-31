// Generated by gencpp from file beginner_tutorials/AddTwoIntsRequest.msg
// DO NOT EDIT!


#ifndef BEGINNER_TUTORIALS_MESSAGE_ADDTWOINTSREQUEST_H
#define BEGINNER_TUTORIALS_MESSAGE_ADDTWOINTSREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace beginner_tutorials
{
template <class ContainerAllocator>
struct AddTwoIntsRequest_
{
  typedef AddTwoIntsRequest_<ContainerAllocator> Type;

  AddTwoIntsRequest_()
    : first()
    , tenth(0.0)  {
    }
  AddTwoIntsRequest_(const ContainerAllocator& _alloc)
    : first(_alloc)
    , tenth(0.0)  {
  (void)_alloc;
    }



   typedef std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other >  _first_type;
  _first_type first;

   typedef double _tenth_type;
  _tenth_type tenth;





  typedef boost::shared_ptr< ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator> const> ConstPtr;

}; // struct AddTwoIntsRequest_

typedef ::beginner_tutorials::AddTwoIntsRequest_<std::allocator<void> > AddTwoIntsRequest;

typedef boost::shared_ptr< ::beginner_tutorials::AddTwoIntsRequest > AddTwoIntsRequestPtr;
typedef boost::shared_ptr< ::beginner_tutorials::AddTwoIntsRequest const> AddTwoIntsRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator1> & lhs, const ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator2> & rhs)
{
  return lhs.first == rhs.first &&
    lhs.tenth == rhs.tenth;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator1> & lhs, const ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace beginner_tutorials

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "20b8b34341923790505b86daa316f2e5";
  }

  static const char* value(const ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x20b8b34341923790ULL;
  static const uint64_t static_value2 = 0x505b86daa316f2e5ULL;
};

template<class ContainerAllocator>
struct DataType< ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "beginner_tutorials/AddTwoIntsRequest";
  }

  static const char* value(const ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "string first\n"
"float64 tenth\n"
;
  }

  static const char* value(const ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.first);
      stream.next(m.tenth);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct AddTwoIntsRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::beginner_tutorials::AddTwoIntsRequest_<ContainerAllocator>& v)
  {
    s << indent << "first: ";
    Printer<std::basic_string<char, std::char_traits<char>, typename ContainerAllocator::template rebind<char>::other > >::stream(s, indent + "  ", v.first);
    s << indent << "tenth: ";
    Printer<double>::stream(s, indent + "  ", v.tenth);
  }
};

} // namespace message_operations
} // namespace ros

#endif // BEGINNER_TUTORIALS_MESSAGE_ADDTWOINTSREQUEST_H
