#include"Util.h"

Util* Util::util = 0;


Util::Util()
{
}


JSON_VALUE Util::getValue(CONFIG con, string key)
{
	const Value& Shape_Obj = doc[toStr(con).c_str()];
	JSON_VALUE jv;
	if (Shape_Obj.IsString())
	{
		jv.json_str_value = Shape_Obj[key.c_str()].GetString();
	}
	else if (Shape_Obj.IsInt())
	{
		jv.json_int_value = Shape_Obj[key.c_str()].GetInt();
	}

	return jv;
}


vector<JSON_VALUE> Util::getValues(CONFIG con)
{
	vector<JSON_VALUE>values_list;
	for (Value::ConstMemberIterator itr = doc[toStr(con).c_str()].MemberBegin(); itr != doc[toStr(con).c_str()].MemberEnd(); ++itr)
	{
		JSON_VALUE jv;
		if (itr->value.IsString())
		{
			jv.json_str_value = itr->value.GetString();
		}
		else if (itr->value.IsInt())
		{
			jv.json_int_value = itr->value.GetInt();
		}
		values_list.push_back(jv);
	}

	return values_list;
}

string Util::toStr(CONFIG con)
{
	string s = "";
	switch (con)
	{
	case INPUT_SHAPE:
		s = "Input_Shape";
		break;
	case TENSOR_SHAPE:
		s = "Tensor_Shape";
		break;
	case BIAS_SHAPE:
		s = "Bias_Shape";
		break;
	case FILTER_SHAPE:
		s = "Filter_Shape";
		break;
	case CONV_SHAPE:
		s = "Conv_Shape";
		break;
	default:
		break;
	}

	return s;
}


Util* Util::getInstance()
{
	if (!util)
	{
		util = new Util();
		return util;
	}
	else
	{
		return util;
	}
}


void Util::read_Json()
{
	config_file = "config.json";
	ifstream json_reader(config_file);
	IStreamWrapper isw(json_reader);
	doc.ParseStream(isw);
}