use ndmp;
drop table rulelab_hierarchical_area;
create table if not exists rulelab_hierarchical_area
(
	id bigint(20) not null auto_increment,
	cname varchar(255) not null comment '地域中文名',
	old_code bigint(20) comment '老国标',
	sng_code bigint(20) comment 'SNG编码',
	gb_code bigint(20) not null comment '新国标',
	gb_code_path varchar(255) comment '父极地域层级信息，包含了父路径信息，分号分割',
	status tinyint(1) default 0 not null comment '0代表active，1代表deprecated',
	deleted tinyint(1) DEFAULT 0 COMMENT '删除标记,0:未删除;1:删除',
  	created_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
  	last_modified_time timestamp(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',
  	revision int(11) NOT NULL DEFAULT 1 COMMENT '修订版本',
  	PRIMARY KEY(id),
  	UNIQUE KEY unique_key(cname,old_code,sng_code,gb_code),
  	key path_key(gb_code_path)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 collate=utf8_bin comment='地址编码层级信息，如果某条rule配置的是深圳市，则深圳市的后代节点都会生效';
