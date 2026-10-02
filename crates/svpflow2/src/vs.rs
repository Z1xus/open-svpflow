pub(crate) use svpflow_host::vs3::*;

pub(crate) trait Source {
    fn source(&self) -> crate::options::SourceInfo;
}

impl Source for VideoInfo {
    fn source(&self) -> crate::options::SourceInfo {
        crate::options::SourceInfo {
            fps_num: self.fps_num,
            fps_den: self.fps_den,
            width: self.width,
            height: self.height,
        }
    }
}
