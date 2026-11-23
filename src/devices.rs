//! Which physical device a video or audio device is part of. Windows gives every part of one
//! device (a capture card's video, its sound input) the same container ID, so the card's sound
//! can be found from its video whatever either is called.

use windows::Win32::Devices::DeviceAndDriverInstallation::{
    CM_Get_DevNode_PropertyW, CM_Get_Device_Interface_PropertyW, CM_LOCATE_DEVNODE_NORMAL, CM_Locate_DevNodeW, CR_SUCCESS,
};
use windows::Win32::Devices::Properties::{DEVPKEY_Device_ContainerId, DEVPKEY_Device_InstanceId, DEVPROPTYPE};
use windows::core::{GUID, HSTRING};

/// The container of the device behind a device interface (a camera's symbolic link).
pub fn container_of_interface(link: &str) -> Option<GUID> {
    let mut buf = [0u16; 512];
    let mut size = std::mem::size_of_val(&buf) as u32;
    let mut kind = DEVPROPTYPE::default();
    let r = unsafe {
        CM_Get_Device_Interface_PropertyW(
            &HSTRING::from(link),
            &DEVPKEY_Device_InstanceId,
            &mut kind,
            Some(buf.as_mut_ptr() as *mut u8),
            &mut size,
            0,
        )
    };
    if r != CR_SUCCESS {
        return None;
    }
    let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    container_of_instance(&String::from_utf16_lossy(&buf[..len]))
}

/// The container of a device instance (`USB\VID_...`, `SWD\MMDEVAPI\{...}` for a sound endpoint).
pub fn container_of_instance(instance: &str) -> Option<GUID> {
    let mut node = 0u32;
    let mut guid = GUID::zeroed();
    let mut size = std::mem::size_of::<GUID>() as u32;
    let mut kind = DEVPROPTYPE::default();
    unsafe {
        if CM_Locate_DevNodeW(&mut node, &HSTRING::from(instance), CM_LOCATE_DEVNODE_NORMAL) != CR_SUCCESS {
            return None;
        }
        if CM_Get_DevNode_PropertyW(
            node,
            &DEVPKEY_Device_ContainerId,
            &mut kind,
            Some(&mut guid as *mut GUID as *mut u8),
            &mut size,
            0,
        ) != CR_SUCCESS
        {
            return None;
        }
    }
    // All-zero and the "no container" ID say nothing about which device it is.
    const NONE: GUID = GUID::from_u128(0x00000000_0000_0000_ffff_ffffffffffff);
    (guid != GUID::zeroed() && guid != NONE).then_some(guid)
}
