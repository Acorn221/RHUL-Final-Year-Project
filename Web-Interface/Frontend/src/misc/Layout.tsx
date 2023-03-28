import { Outlet } from 'react-router';

const Layout = () => (
  <div className="flex justify-center align-middle h-screen">
    <Outlet />
  </div>
);

export default Layout;
