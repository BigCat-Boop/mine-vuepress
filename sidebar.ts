import dirTree, { DirectoryTree } from 'directory-tree';

interface SidebarOption {
  text: string;
  collapsible?: boolean;
  collapsed?: boolean;
  items?: SidebarOption[];
  link?: string;
}

function toSidebarOption(tree: DirectoryTree[]): SidebarOption[] {
  if (!Array.isArray(tree)) return [];

  return tree.map((v) => {
    if (v.children !== undefined) {
      return {
        text: v.name,
        collapsible: true,
        collapsed: true,
        items: toSidebarOption(v.children),
      };
    } else {
      return {
        text: v.name.replace('.md', ''),
        link: v.path.split('mine-vuepress')[1].replace('.md', ''),
      };
    }
  });
}

function autoGetSidebarOptionBySrcDir(srcPath: string, title?: string): SidebarOption[] {
  const srcDir = dirTree(srcPath, {
    extensions: /\.md$/,
    normalizePath: true,
  });

  if (!srcDir || !srcDir.children) return [];

  return [
    {
      text: title ?? srcDir.name,
      collapsible: true,
      collapsed: true,
      items: toSidebarOption(srcDir.children),
    },
  ];
}

export default autoGetSidebarOptionBySrcDir;
